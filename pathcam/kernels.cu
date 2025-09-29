#include "cuda_runtime.h"
#include <opencv2/opencv.hpp>


__global__ void drop_alpha_and_swap(char *dst, char *src, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    const char *src_pixel = src + idx * 4;
    char *dst_pixel = dst + idx * 3;

    dst_pixel[0] = src_pixel[2]; // swap B ↔ R
    dst_pixel[1] = src_pixel[1];
    dst_pixel[2] = src_pixel[0];
  }
}

extern "C" void launch_drop_alpha_and_swap(char *dst, char *src, int count) {
  int threadsPerBlock = 256;
  int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
  drop_alpha_and_swap<<<blocks, threadsPerBlock>>>(dst, src, count);
  cudaDeviceSynchronize(); // optional, for sync
}

__global__ void make_hann_1d(float *w, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    // Hann: 0.5 * (1 - cos(2*pi*i/(n-1)))
    float t = (n > 1) ? (float) i / (float) (n - 1) : 0.0f;
    w[i] = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * t));
  }
}

extern "C" void ensure1DHann(int w, int h, cv::cuda::GpuMat &wx, cv::cuda::GpuMat &wy, cudaStream_t stream) {
  if (wx.cols != w || wx.rows != 1 || wx.type() != CV_32F) wx.create(1, w, CV_32F);
  if (wy.cols != 1 || wy.rows != h || wy.type() != CV_32F) wy.create(h, 1, CV_32F);

  dim3 block(256, 1);
  dim3 gridX((w + block.x - 1) / block.x, 1);
  dim3 gridY((h + block.x - 1) / block.x, 1);

  make_hann_1d<<<gridX, block, 0, stream>>>(wx.ptr<float>(), w);
  make_hann_1d<<<gridY, block, 0, stream>>>(wy.ptr<float>(), h);
}

__global__ void apply_hann_2d(const float *__restrict__ src, size_t srcStep,
                              float *__restrict__ dst, size_t dstStep,
                              const float *__restrict__ wx,
                              const float *__restrict__ wy,
                              int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x; // column
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row
  if (x < w && y < h) {
    const float *srcRow = (const float *) ((const uchar *) src + y * srcStep);
    float *dstRow = (float *) ((uchar *) dst + y * dstStep);
    dstRow[x] = srcRow[x] * wx[x] * wy[y];
  }
}

extern "C" void launch_apply_hann_2d(cv::cuda::GpuMat &wx, cv::cuda::GpuMat &wy, cv::cuda::GpuMat &win,
                                     cv::cuda::GpuMat &magNorm, cudaStream_t stream) {
  dim3 blk(32, 16);
  dim3 grd((magNorm.cols + blk.x - 1) / blk.x,
           (magNorm.rows + blk.y - 1) / blk.y);
  apply_hann_2d<<<grd, blk, 0, stream>>>(
    magNorm.ptr<float>(), magNorm.step,
    win.ptr<float>(), win.step,
    wx.ptr<float>(), wy.ptr<float>(),
    magNorm.cols, magNorm.rows
  );
}

__global__ void cross_power_spectrum(const float2 *__restrict__ F, size_t Fstep,
                                     const float2 *__restrict__ G, size_t Gstep,
                                     float2 *__restrict__ CPS, size_t CPSstep,
                                     int w, int h, float eps) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    const float2 *Frow = (const float2 *) ((const uchar *) F + y * Fstep);
    const float2 *Grow = (const float2 *) ((const uchar *) G + y * Gstep);
    float2 *CPSrow = (float2 *) ((uchar *) CPS + y * CPSstep);

    float2 a = Frow[x];
    float2 b = Grow[x]; // conj(b) = (b.x, -b.y)

    // complex multiply: a * conj(b)
    float2 c;
    c.x = a.x * b.x + a.y * b.y; // real
    c.y = a.y * b.x - a.x * b.y; // imag

    float mag = sqrtf(c.x * c.x + c.y * c.y);
    float inv = 1.0f / (mag + eps);

    CPSrow[x].x = c.x * inv;
    CPSrow[x].y = c.y * inv;
  }
}

extern "C" void launch_CPS(const cv::cuda::GpuMat &F,
                           const cv::cuda::GpuMat &G,
                           cv::cuda::GpuMat &CPS,
                           float eps = 1e-9f,
                           cudaStream_t stream) {
  const int w = F.cols;
  const int h = F.rows;

  // Grid/block
  dim3 block(32, 16);
  dim3 grid((w + block.x - 1) / block.x,
            (h + block.y - 1) / block.y);

  cross_power_spectrum<<<grid, block, 0, stream>>>(F.ptr<float2>(), F.step, G.ptr<float2>(), G.step, CPS.ptr<float2>(),
                                                   CPS.step, w, h, eps);
}
