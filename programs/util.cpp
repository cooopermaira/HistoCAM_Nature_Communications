//
// Created by cooper maira on 10/9/25.
//

//#include "pathCam.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>

cv::Mat makeMotionKernel(int length, float angleDeg)
{
  // Ensure length is at least 1 and odd
  length = std::max(1, length);
  if (length % 2 == 0) length += 1;

  int ksize = length; // you can pad larger if you want, but this is enough
  cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);

  // Draw a horizontal line (center row) of "length" ones
  int center = ksize / 2;
  int half = length / 2;
  for (int x = center - half; x <= center + half; ++x)
    kernel.at<float>(center, x) = 1.0f;

  // Normalize to sum = 1
  kernel /= static_cast<float>(length);

  // Rotate the kernel by angleDeg around its center
  cv::Point2f rotCenter(ksize / 2.0f, ksize / 2.0f);
  cv::Mat rotMat = cv::getRotationMatrix2D(rotCenter, angleDeg, 1.0);

  cv::Mat rotated;
  cv::warpAffine(kernel, rotated, rotMat, kernel.size(),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

  // Re-normalize (rotation/interpolation slightly changes sum)
  double sum = cv::sum(rotated)[0];
  if (sum != 0.0)
    rotated /= static_cast<float>(sum);

  return rotated;
}
bool loadRawToGpuGray(const std::string &path,
                             int width, int height,
                             cv::cuda::GpuMat &outGray,
                             int bayerCode = cv::COLOR_BayerBG2GRAY) {
  // Read file
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) {
    std::cerr << "[WARN] Cannot open file: " << path << "\n";
    return false;
  }
  std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);

  const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (size != static_cast<std::streamsize>(expected)) {
    std::cerr << "[WARN] Size mismatch for " << path
        << " (got " << size << ", expected " << expected << " bytes)\n";
    return false;
  }

  std::vector<unsigned char> buf(expected);
  if (!f.read(reinterpret_cast<char *>(buf.data()), size)) {
    std::cerr << "[WARN] Failed to read data: " << path << "\n";
    return false;
  }

  // Wrap CPU Mat as single-channel 8-bit
  cv::Mat raw(height, width, CV_8UC1, buf.data());

  // Upload to GPU
  cv::cuda::GpuMat dRaw, dGray;
  dRaw.upload(raw);

  // Debayer on GPU -> grayscale
  cv::cuda::cvtColor(dRaw, dGray, bayerCode);

  // Return
  outGray = dGray; // shallow copy of GpuMat header (data stays on GPU)
  return true;
}