
#include "cuda_runtime.h"

__global__ void drop_alpha_and_swap(char* dst, char* src,int count) {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx < count) {
        const char* src_pixel = src + idx * 4;
        char* dst_pixel = dst + idx * 3;

        dst_pixel[0] = src_pixel[2];  // swap B ↔ R
        dst_pixel[1] = src_pixel[1];
        dst_pixel[2] = src_pixel[0];
      }
    }

extern "C" void launch_drop_alpha_and_swap(char* dst, char* src, int count) {
    int threadsPerBlock = 256;
    int blocks = (count + threadsPerBlock - 1) / threadsPerBlock;
    drop_alpha_and_swap<<<blocks, threadsPerBlock>>>(dst, src, count);
    cudaDeviceSynchronize();  // optional, for sync
}