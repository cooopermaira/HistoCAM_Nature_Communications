//
// Created by cooper maira on 10/9/25.
//
#include "Poco/DirectoryIterator.h"
#include <regex>
#include <fstream>
#include <cstdio>
#include <string>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

extern "C" void thresholdAndPackToBits(const cv::cuda::GpuMat& gray8u,
                              double percentile,
                              cv::cuda::GpuMat& bitmaskBytes);

Mat makeMotionKernel(int length, float angleDeg);

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


  std::vector<cuda::GpuMat> files;
  std::vector<std::string> paths;
  const std::string path = argv[1];
  for (Poco::DirectoryIterator it(path), end; it != end; ++it) {
    if (!it->isFile()) continue;
    const std::string p = it->path();
    if (hasRawExtension(p)) {
      paths.push_back(p);
    }
  }

  Size image_size(6464,4852);
  Mat img,raw;
  char *bufHost = new char[image_size.width * image_size.height];
  for (auto &path : paths) {

    std::ifstream stream;
    stream.open(path, std::ios::binary);
    stream.read(bufHost, image_size.width * image_size.height);
    stream.close();

    raw = Mat(image_size,CV_8UC1,bufHost);
    cvtColor(raw,img,COLOR_BayerBG2BGR);
    Mat test;
    resize(img,test,Size(image_size.width / 10,image_size.height/10));

    int k = 0;
  }



}