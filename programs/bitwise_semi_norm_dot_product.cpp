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



int main(int argc, char *argv[]) {
  std::string blurPath = "/home/pathcam/pcamdata/postProc/blur_test/raw/23.Raw";
  std::string sharpPath = "/home/pathcam/pcamdata/postProc/blur_test/raw/22.Raw";

  Size image_size(6464,4852);

  auto blurRaw = load_raw(blurPath);
  char color = 'R';
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
  //imwrite("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/kernel_"+std::to_string(ksize)+"_"+color+".tiff",kern);
  auto val = imread("/home/pathcam/pcamdata/postProc/blur_test/train/misc/out/kernel_17_R.tiff",IMREAD_UNCHANGED);
  auto ret = sum(val);
  //normalize(val,val,0,255,NORM_MINMAX,CV_8U);
  int k = 0;



}