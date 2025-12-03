//
// Created by cooper maira on 10/9/25.
//
// #include "Poco/DirectoryIterator.h"
#include <regex>
// #include <fstream>
// #include <cstdio>
// #include <string>
// #include <opencv2/core/cuda.hpp>
// #include <opencv2/opencv.hpp>
#include "pathCam.h"
using namespace cv;
enum BayerPattern { RGGB, BGGR, GBRG, GRBG };

extern "C" void thresholdAndPackToBits(const cv::cuda::GpuMat& gray8u,
                              double percentile,
                              cv::cuda::GpuMat& bitmaskBytes);

Mat makeMotionKernel(int length, float angleDeg);
Mat load_raw(const std::string &path);
void find_best_matching_roi(Mat &rawBlur, Mat &rawSharp,int patchSize);
std::vector<cv::Mat> extractDenseBayerChannels(const cv::Mat &bayer, BayerPattern pattern);
// Return 0=R, 1=G, 2=R for given Bayer coordinate under BGGR
cv::Mat estimateKernelFromBayerSingleColor(
    const cv::Mat &sharpBayer,
    const cv::Mat &targetDense,
    int ksize,
    char colorChar,
    BayerPattern pattern = BGGR,
    double lambda = 1e-6);
cv::Mat estimateKernel_CGLS(const cv::Mat &sharpBayer,   // CV_32F, 1ch, padded: Hb x Wb
    const cv::Mat &targetDense,  // CV_32F, 1ch: Hd x Wd
    int ksize,
    char colorChar,
    BayerPattern pattern = BGGR,
    double lambda = 1e-6,
    int maxIters = 25,
    double tol = 1e-6
);
cv::Mat convolveBayerToDenseSingleChannel_F32(
    const cv::Mat &bayer,
    const cv::Mat &kernel,
    char colorFlag);
bool loadRawToGpuGray(const std::string &path,
                             int width, int height,
                             cv::cuda::GpuMat &outGray,
                             int bayerCode = cv::COLOR_BayerBG2GRAY);
static bool hasRawExtension(const std::string &name) {
  auto pos = name.find_last_of('.');
  if (pos == std::string::npos) return false;
  std::string ext = name.substr(pos + 1);
  for (auto &c: ext) c = (char) std::tolower((unsigned char) c);
  return ext == "raw"; // accept ".raw" (case-insensitive)
}
int extractMagnification(const std::string &path) {
  std::regex re("([0-9]+)x");
  std::smatch match;
  if (std::regex_search(path, match, re)) {
    return std::stoi(match[1].str());
  }
  return -1; // fallback if no match
}
void projectKernelNonnegSum1(cv::Mat &k)
{
  CV_Assert(k.type() == CV_32F);

  // 1) Clamp negatives to 0
  cv::max(k, 0.0f, k);  // k = max(k, 0)

  // // 2) Compute sum
  // double sumK = cv::sum(k)[0];
  //
  // // 3) Avoid division by zero: if all zeros, just put a delta at center
  // if (sumK <= 1e-12) {
  //   k.setTo(0.0f);
  //   cv::Point center(k.cols / 2, k.rows / 2);
  //   k.at<float>(center) = 1.0f;
  //   return;
  // }
  //
  // // 4) Normalize
  // k /= static_cast<float>(sumK);
}
void applyAlphaToBayer_BGGR(cv::Mat& img32f, const cv::Vec3f& alphaBGR);
cv::Vec3f computePerChannelScale_BGGR(const cv::Mat &imgA_in,
                                      const cv::Mat &imgB_in);
Mat convolveBayerWithKernels_BGGR(
    const cv::Mat &rawBayerIn,
    const std::vector<cv::Mat> &kernels  // 3 kernels, CV_32F, same size
);
float computeGlobalAlpha(const cv::Mat &A_in, const cv::Mat &B_in);

int main(int argc, char *argv[]) {
  std::string sharpPath = "/home/pathcam/pcamdata/postProc/blur_test/raw/22.Raw";
  auto sharpRaw = load_raw(sharpPath);


  std::vector<Mat> kernelsBGR;
  char colors[] = {'B', 'G', 'R'};
  int patchSize = 2000;


  for (int i = 0; i < 3; ++i) {
    std::string path = "/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/kernel_17_";
    path.push_back(colors[i]);
    path += ".tiff";

    kernelsBGR.push_back(imread(path, IMREAD_UNCHANGED));
    projectKernelNonnegSum1(kernelsBGR[i]);
  }

  int ksize = kernelsBGR[0].rows;
  pathCam::Image image(6464 - ksize + 1,4852- ksize + 1,2040);
  image.set_disk_file("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/22_convolved_by_channel.Raw");
  Rect sharpRect((sharpRaw.cols - patchSize) / 2, (sharpRaw.rows - patchSize) / 2,patchSize,patchSize);
  Rect blurRect((image.width - patchSize) / 2, (image.height - patchSize) / 2, patchSize,patchSize);

  auto img = convolveBayerWithKernels_BGGR(sharpRaw,kernelsBGR);

   auto alphas = computePerChannelScale_BGGR(img(blurRect),sharpRaw(sharpRect));
  for (int i = 0; i < kernelsBGR.size(); ++i) {
    kernelsBGR[i] *= alphas[i];
  }


  img = convolveBayerWithKernels_BGGR(sharpRaw,kernelsBGR);

  Mat rcv;
  img.convertTo(img,CV_8U);
  cvtColor(img,rcv,COLOR_BayerBG2BGR);
  imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/22_convolved_by_channel_alpha_adjust_kernels.png",rcv);
  Rect roi(2416 - ksize/2,1422 - ksize/2,image.blurPatch,image.blurPatch);

  //imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/22_convolved_by_channel_dft_uhoh.png",image.blurDFT);
  // assert(img.isContinuous());
  // size_t t = img.total(),e=img.elemSize();
  // assert(t * e == image.width * image.height);
  image.copy_in(img.data);
  image.check_blur(true);
  imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/22_convolved_by_channel_dft_sum1.png",image.blurDFT);
  //image.write_to_path();
  return 0;

  image.load_raw_from_disk();
  Size image_size(image.width, image.height);
  Mat img2(image_size,CV_8U,image.get_Raw());
  //cvtColor(img2,img2,COLOR_BayerBG2BGR);
  //imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/22_convolved_by_channel.png",img2);
  image.check_blur(true);
  auto m = image.blurDFT;
  imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/22_convolved_by_channel_dft.png",m);

  image.free_memory_RAW();


  int k = 0;

}

int main1(int argc, char *argv[]) {
  std::string blurPath = "/home/pathcam/pcamdata/postProc/blur_test/raw/23.Raw";
  std::string sharpPath = "/home/pathcam/pcamdata/postProc/blur_test/raw/22.Raw";

  Size image_size(6464,4852);

  auto blurRaw = load_raw(blurPath);
  char color = 'G';
  int patchSize = 2000;
  int ksize = 17;
  Rect blurRect((blurRaw.cols - patchSize) / 2,(blurRaw.rows - patchSize) / 2,patchSize,patchSize);
  auto sharpRaw = load_raw(sharpPath);
  Rect sharpRect(2416 - ksize/2,1422 - ksize/2,patchSize+ksize - 1,patchSize+ksize - 1);
  //imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/clearframe.png",sharpRaw);

  //find_best_matching_roi(blurRaw,sharpRaw,2000);
  Mat source, target;
  auto chn = extractDenseBayerChannels(blurRaw(blurRect),BGGR);
  //imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/clearPatch.png",sharpRaw(sharpRect));


  int chnInd = color == 'R' ? 0 : color == 'G' ? 1 : 2;
  chn[chnInd].convertTo(target,CV_32F);
  // pathCam::Image image(6464,4852,2040);
  // image.set_disk_file("/home/pathcam/pcamdata/postProc/blur_test/raw/23.Raw");
  // image.load_raw_from_disk();
  //
  // Rect roi(0,0,image.blurPatch,image.blurPatch);
  //
  // //image.check_blur(true,chn[1](roi));
  // image.check_blur(true);
  // auto dft = image.blurDFT;



  sharpRaw(sharpRect).convertTo(source,CV_32F);
  auto kern = estimateKernel_CGLS(source,target,ksize,color,BGGR,1e-5,70,1e-12);
  imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/kernel_"+std::to_string(ksize)+"_"+color+".tiff",kern);
  auto val = imread("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/kernel_17_R.tiff",IMREAD_UNCHANGED);
  auto ret = sum(val);
  //normalize(val,val,0,255,NORM_MINMAX,CV_8U);
  int k = 0;



}