//
// Created by cooper maira on 10/9/25.
//
#include "Poco/DirectoryIterator.h"
#include <regex>
#include <cstdio>
#include <string>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

extern "C" void thresholdAndPackToBits(const cv::cuda::GpuMat& gray8u,
                              double percentile,
                              cv::cuda::GpuMat& bitmaskBytes);


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

float test_ratios(std::vector<cuda::GpuMat> files,std::vector<float> ratios) {
  assert(files.size() == 2);

  float
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

  if (true) {
    std::sort(paths.begin(), paths.end(), [](const std::string &a, const std::string &b) {
      return extractMagnification(a) < extractMagnification(b);
    });
  }
  for (auto &p: paths) {
    cuda::GpuMat im;
    loadRawToGpuGray(p, 6464, 4852, im);
    files.push_back(im);
  }



}