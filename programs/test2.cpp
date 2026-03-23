//
// Created by cooper maira on 3/23/26.
//

#include <fstream>
#include <opencv2/imgproc.hpp>

#include "opencv2/core.hpp"
int width = 6464, height = 4852;
using namespace cv;

struct PhaseScaleResult {
  double scale;
  cv::Point2d shift;
  double response;
  bool valid;
};

Mat load_raw(std::string path) {
  char *buf = (char *) malloc(height * width);
  std::ifstream stream;
  stream.open(path, std::ios::binary);
  stream.read(buf, width * height);
  stream.close();
  Mat raw(height, width,CV_8UC1, buf);
  Rect roi(3000, 2000, 500, 500);
  auto v = raw(roi);
  cvtColor(raw, raw, COLOR_BayerBG2GRAY);
  return raw;
}

PhaseScaleResult estimate_scale_phase(
  const cv::Mat &root,
  const cv::Mat &target,
  const std::vector<double> &candidateScales) {
  PhaseScaleResult best{0, {}, -1.0, false};

  cv::Mat root32f, target32f;
  root.convertTo(root32f, CV_32F);
  target.convertTo(target32f, CV_32F);

  // downsample for speed
  const int maxDim = 512;
  auto downscale = [&](const cv::Mat &in) {
    double scale = std::min(1.0, maxDim / double(std::max(in.cols, in.rows)));
    cv::Mat out;
    cv::resize(in, out, {}, scale, scale, cv::INTER_AREA);
    return out;
  };

  root32f = downscale(root32f);
  target32f = downscale(target32f);

  // normalize
  auto normalize = [](cv::Mat &m) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(m, mean, stddev);
    m = (m - mean[0]) / (stddev[0] + 1e-6);
  };

  normalize(root32f);
  normalize(target32f);

  // windowing (critical)
  cv::Mat hann;
  cv::createHanningWindow(hann, root32f.size(), CV_32F);

  root32f = root32f.mul(hann);

  std::vector<double> offsets = {0.95, 0.975, 1.0, 1.025, 1.05};

  double bestResponse = -1;
  cv::Point2d bestShift;
  double bestScale = 0;

  int searchRadius = 300;
  int step = 64;

  for (double scale: candidateScales) {
    cv::Mat scaled;
    cv::resize(target, scaled, {}, scale, scale, cv::INTER_LINEAR);

    for (int dx = -searchRadius; dx <= searchRadius; dx += step) {
      for (int dy = -searchRadius; dy <= searchRadius; dy += step) {
        int x = (scaled.cols - root.cols) / 2 + dx;
        int y = (scaled.rows - root.rows) / 2 + dy;

        if (x < 0 || y < 0 ||
            x + root.cols > scaled.cols ||
            y + root.rows > scaled.rows)
          continue;

        cv::Mat roi = scaled(cv::Rect(x, y, root.cols, root.rows));

        double response;
        cv::Point2d shift = cv::phaseCorrelate(root, roi, cv::noArray(), &response);

        if (response > bestResponse) {
          bestResponse = response;
          bestShift = shift + cv::Point2d(dx, dy);
          bestScale = scale;
        }
      }
    }
  }

  return best;
}

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

struct HomographyResult {
  cv::Mat H; // 3x3 homography
  std::vector<char> inliers; // mask from RANSAC
  int numInliers = 0;
  bool valid = false;
  float scale;
};
struct HomographyResultM {
  cv::Mat H;
  float scale = 1.0f;
  int inliers = 0;
  bool valid = false;
};
HomographyResultM findHomographyAKAZE_multiscale(
    const cv::Mat& img1_gray,
    const cv::Mat& img2_gray,
    const std::vector<float>& scales)
{
  float baseScale = 0.5; // or even 0.2
  Mat img1_small, img2_small;
  resize(img1_gray, img1_small, Size(), baseScale, baseScale);
  resize(img2_gray, img2_small, Size(), baseScale, baseScale);
  HomographyResultM best;
  auto akaze = cv::AKAZE::create();
  std::vector<cv::KeyPoint> kp1;
  Mat desc1;
  akaze->detectAndCompute(img1_small, cv::noArray(), kp1, desc1);
  for (float s : scales){
    // --- 1. Rescale img2 to match img1 scale ---
    cv::Mat img2_scaled;
    cv::resize(img2_small, img2_scaled, cv::Size(), s, s, cv::INTER_LINEAR);



    // --- 2. Detect + compute ---



    cv::Mat desc2;

    std::vector<cv::KeyPoint> kp2;
    akaze->detectAndCompute(img2_scaled, cv::noArray(), kp2, desc2);

    if (desc1.empty() || desc2.empty())
      continue;

    // --- 3. Match ---
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc1, desc2, knn, 2);

    std::vector<cv::DMatch> good;
    for (auto& m : knn)
    {
      if (m.size() < 2) continue;
      if (m[0].distance < 0.75f * m[1].distance)
        good.push_back(m[0]);
    }

    if (good.size() < 10)
      continue;

    // --- 4. Points ---
    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : good)
    {
      pts1.push_back(kp1[m.queryIdx].pt);
      pts2.push_back(kp2[m.trainIdx].pt);
    }

    // --- 5. Homography ---
    std::vector<char> mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, mask);

    if (H.empty())
      continue;

    int inliers = std::count(mask.begin(), mask.end(), 1);

    if (inliers > best.inliers)
    {
      best.H = H;
      best.scale = s;
      best.inliers = inliers;
      best.valid = true;
    }
    kp2.clear();
  }

  return best;
}


int main(int argc, char **argv) {
  auto root = load_raw("/home/cm/Documents/data/phase_corr_testing/test_images_raw/breast/B1_2x.raw");
  auto target = load_raw("/home/cm/Documents/data/phase_corr_testing/test_images_raw/breast/B1_40x.raw");
  std::vector<float> cand10 = {5,2.5,1,0.5,0.25};
  std::vector<float> candidateScale = {1, 0.5, 0.2, 0.1, 0.05};
  auto start = std::chrono::high_resolution_clock::now();

  auto res = findHomographyAKAZE_multiscale(root, target,candidateScale);

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::high_resolution_clock::now() - start).count();
  if (res.valid) {
    // std::cout << "Inliers: " << res.numInliers << std::endl;
    std::cout << "H:\n" << res.H << std::endl;
    std::cout<<"scale "<<res.scale<<std::endl;
    std::cout<<duration<<std::endl;
  }
  return 0;
};
