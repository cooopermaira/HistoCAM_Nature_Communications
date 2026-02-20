#include <fstream>
#include <sstream>
#include <iomanip>
#include <bits/regex_error.h>

#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include "opencv2/core.hpp"

using namespace cv;

cv::Mat hann;

std::pair<float, float> meanStdDev(const std::vector<float>& values)
{
  const size_t n = values.size();
  if (n == 0)
    return {0.0f, 0.0f};

  float mean = 0.0f;
  float M2   = 0.0f;
  size_t k   = 0;

  for (float x : values)
  {
    ++k;
    float delta  = x - mean;
    mean        += delta / k;
    float delta2 = x - mean;
    M2          += delta * delta2;
  }

  float variance = (n > 1) ? (M2 / (n - 1)) : 0.0f;  // sample variance
  float stddev   = std::sqrt(variance);

  return {mean, stddev};
}

float estimateAdditionalGaussianBlurSigma(
  const cv::Mat &sharp,
  const cv::Mat &blurry,
  float rmin_frac = 0.05f, // ignore very low frequencies
  float rmax_frac = 0.5f // ignore extreme high frequencies
) {
  CV_Assert(sharp.type() == CV_32FC1);
  CV_Assert(blurry.type() == CV_32FC1);
  CV_Assert(sharp.size() == blurry.size());

  const int rows = sharp.rows;
  const int cols = sharp.cols;

  const float eps = 1e-8f;

  // --------------------------------------------
  // 1) Mean removal
  // --------------------------------------------
  cv::Mat s = sharp - cv::mean(sharp)[0];
  cv::Mat b = blurry - cv::mean(blurry)[0];

  s = s.mul(hann);
  b = b.mul(hann);

  // --------------------------------------------
  // 3) Compute DFT
  // --------------------------------------------
  cv::Mat FS, FB;
  cv::dft(s, FS, cv::DFT_COMPLEX_OUTPUT);
  cv::dft(b, FB, cv::DFT_COMPLEX_OUTPUT);

  // Split real/imag
  std::vector<cv::Mat> planesS, planesB;
  cv::split(FS, planesS);
  cv::split(FB, planesB);

  cv::Mat magS, magB;
  cv::magnitude(planesS[0], planesS[1], magS);
  cv::magnitude(planesB[0], planesB[1], magB);

  // --------------------------------------------
  // 4) Radial regression accumulation
  //    Fit: log(R) = alpha * f^2
  // --------------------------------------------

  float sum_xx = 0.0f;
  float sum_xy = 0.0f;

  const float cx = cols / 2.0f;
  const float cy = rows / 2.0f;
  const float nyquist = std::sqrt(cx * cx + cy * cy);

  const float rmin = rmin_frac * nyquist;
  const float rmax = rmax_frac * nyquist;

  for (int y = 0; y < rows; ++y) {
    int fy = (y <= rows / 2) ? y : y - rows;

    for (int x = 0; x < cols; ++x) {
      int fx = (x <= cols / 2) ? x : x - cols;

      float r = std::sqrt(float(fx * fx + fy * fy));
      if (r < rmin || r > rmax)
        continue;

      float mS = magS.at<float>(y, x);
      if (mS < eps)
        continue;

      float mB = magB.at<float>(y, x);

      float R = mB / (mS + eps);

      // Clamp to [0,1] (blur shouldn't amplify magnitude)
      if (R > 1.0f)
        R = 1.0f;

      if (R <= 0.0f)
        continue;

      float y_val = std::log(R);
      float x_val = r * r;

      sum_xx += x_val * x_val;
      sum_xy += x_val * y_val;
    }
  }

  if (sum_xx < eps)
    return 0.0f;

  float alpha = sum_xy / sum_xx; // slope

  if (alpha >= 0.0f)
    return 0.0f; // no additional blur detected

  float sigma = std::sqrt(-alpha / (2.0f * float(M_PI) * float(M_PI)));

  return sigma;
}

Mat log_mag_dft(Mat img) {
  assert(img.rows == hann.rows && img.cols == hann.cols && img.depth() == CV_32F && img.channels() == 1);
  multiply(img,hann,img);
  Mat planes[] = {img,Mat(img.rows,img.cols,CV_32F,Scalar(0))};
  Mat complexI,mag;
  merge(planes,2,complexI);

  dft(complexI,complexI);

  split(complexI,planes);
  magnitude(planes[0],planes[1],mag);
  add(Scalar(1e-6),mag,mag);
  log(mag,mag);

  GaussianBlur(mag,mag,Size(3,3),3,3);
  Mat res;
  normalize(mag,res, 0,255,NORM_MINMAX,CV_8U);
  return res;
}

std::pair<float, float> run_for_img_pair(const Mat& blurMat, const Mat& sharpMat,int roiSize) {
  int increment = 20;
  std::vector<float> scores;
  Mat display;
  for (int x = 0; x < blurMat.cols - roiSize; x += increment) {
    for (int y = 0; y < blurMat.rows - roiSize; y += increment) {
      Rect roi(x, y, roiSize, roiSize);
      auto blurRoi = blurMat(roi);
      auto sharpRoi = sharpMat(roi);
      // auto ans = estimateAdditionalGaussianBlurSigma(sharpRoi, blurRoi);
      // scores.push_back(ans);
      blurRoi.convertTo(display,CV_8U);
      auto dft = log_mag_dft(blurRoi);

      int k = 0;
    }
  }
  return meanStdDev(scores);
}


int main(int argc, char **argv) {
  int roiSize = 1 * 128;
  cv::createHanningWindow(hann, Size(roiSize, roiSize), CV_32F);

  std::string file = argv[1];
  for (int i = 1; i < 2900; i+=30) {
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << i;
    std::string fileName = oss.str()+".png";

    std::string blurImg = file + "/blur/images/" + fileName;
    std::string sharpImg = file + "/sharp/images/" + fileName;

    Mat blurMat = imread(blurImg);
    Mat sharpMat = imread(sharpImg);

    cvtColor(blurMat, blurMat, COLOR_BGR2GRAY);
    cvtColor(sharpMat, sharpMat, COLOR_BGR2GRAY);

    blurMat.convertTo(blurMat,CV_32FC1);
    sharpMat.convertTo(sharpMat,CV_32FC1);

    auto [mean,stddv] = run_for_img_pair(blurMat,sharpMat,roiSize);
    std::cout<<fileName + " "<<mean<<" "<<stddv<<std::endl;
  }






  // std::cout<<mean<<" "<<stddv<<std::endl;
}
