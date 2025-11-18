//
// Created by cooper maira on 10/9/25.
//
//#include "pathCam.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

// ---- CUDA kernel: pack 8 binary pixels (0/255) into one byte (LSB-first) ----
__global__ void packBitsKernel(const unsigned char* __restrict__ src,
                               size_t src_step, int rows, int cols,
                               unsigned char* __restrict__ dst,
                               size_t dst_step)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int xByte = blockIdx.x * blockDim.x + threadIdx.x; // byte column index

    if (y >= rows) return;

    int nBytes = (cols + 7) >> 3;
    if (xByte >= nBytes) return;

    const unsigned char* srcRow = src + y * src_step;
    unsigned char* dstRow = dst + y * dst_step;

    unsigned char b = 0;
    int base = xByte << 3;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int x = base + k;
        if (x < cols) {
            // LSB corresponds to the leftmost pixel in this 8-pixel group.
            b |= (srcRow[x] != 0) << k;
        }
    }
    dstRow[xByte] = b;
}

// ---- Compute percentile threshold (0..100) for CV_8UC1 GpuMat ----
// Uses 256-bin histogram on GPU, downloads 256 ints to CPU, finds cutoff.
int percentileThresholdGPU(const cv::cuda::GpuMat& gray8u,
                           double percentile)
{
    CV_Assert(gray8u.type() == CV_8UC1);
    CV_Assert(percentile >= 0.0 && percentile <= 100.0);

    // Compute histogram on GPU
    cv::cuda::GpuMat d_hist; // 256x1 histogram
    cv::cuda::calcHist(gray8u, d_hist);

    // Download the small 256-bin hist to host
    cv::Mat h_hist;
    d_hist.download(h_hist);

    // Accumulate to find the percentile bin
    using int64 = long long;
    int64 total = static_cast<int64>(gray8u.rows) * gray8u.cols;
    int64 target = static_cast<int64>(std::llround((percentile / 100.0) * total));
    target = std::min(std::max(target, int64(0)), total);

    const int* bins = h_hist.ptr<int>();
    int64 cum = 0;
    int thresh = 255; // fallback
    for (int v = 0; v < 256; ++v) {
        cum += bins[v];
        if (cum >= target) { thresh = v; break; }
    }
    return thresh;
}

// ---- Main routine: threshold at percentile and pack to bitmask ----
// Output bitmask has size rows x ceil(cols/8) of type CV_8UC1.
extern "C" void thresholdAndPackToBits(const cv::cuda::GpuMat& gray8u,
                            double percentile,
                            cv::cuda::GpuMat& bitmaskBytes)
{
    CV_Assert(gray8u.type() == CV_8UC1);

    // 1) Find threshold value at percentile
    int thr = percentileThresholdGPU(gray8u, percentile);

    // 2) Binary threshold on GPU: > thr -> 255, else 0
    char* databin = new char[gray8u.rows * gray8u.cols];
    cv::cuda::GpuMat d_bin(gray8u.rows,gray8u.cols,CV_8UC1,databin);
    cv::cuda::threshold(gray8u, d_bin, thr, 255, cv::THRESH_BINARY);
    // cv::Mat temp;
    // d_bin.download(temp);
    // cv::imwrite("/media/max/Data/phase_corr_test/thresholded_liver"+std::to_string(int(percentile))+".png",temp);

    // 3) Allocate output bitmask: width in bytes
    int rows = d_bin.rows;
    int cols = d_bin.cols;
    int outColsBytes = (cols + 7) >> 3;
    char* data = new char[rows * outColsBytes];
    bitmaskBytes = cv::cuda::GpuMat(rows, outColsBytes, CV_8UC1,data);

    // 4) Launch pack kernel on same stream
    dim3 block(32, 16);
    dim3 grid((outColsBytes + block.x - 1) / block.x,
              (rows        + block.y - 1) / block.y);

    packBitsKernel<<<grid, block, 0>>>(
        d_bin.ptr<unsigned char>(), d_bin.step, rows, cols,
        bitmaskBytes.ptr<unsigned char>(), bitmaskBytes.step
    );

}
