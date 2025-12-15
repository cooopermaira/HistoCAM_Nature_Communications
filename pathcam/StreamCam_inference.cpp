//
// Created by Cooper Maira on 12/10/25.
//

#include "pathCam.h"

int maxBatchSize = 64;

namespace pathCam {
  using namespace nvinfer1;

  void StreamCam::process_blur_Q() {
    while (compositing) {
      launch_blur_metric();
      receive_blur_metric();
    }
  }

  void StreamCam::clean_up_blur_engine() const {
    delete blurEngine;
    cudaStreamDestroy(blurStream);
    cudaFree(blurInputs);
  }

  void StreamCam::Q_blur_metric(Image *image) {
    blurMutex.lock();
    blurMeticQ.push(image);
    blurMutex.unlock();
  }

  void StreamCam::load_blur_engine() {
    auto dBlob = readFile(no_ref_blur_model_path.toString());

    IRuntime *bRuntime = createInferRuntime(nvloger);
    blurEngine = bRuntime->deserializeCudaEngine(dBlob.data(), dBlob.size());
    delete bRuntime;

    assert(blurEngine);
    blurCtx = blurEngine->createExecutionContext();
    cudaStreamCreate(&blurStream);

    CHECK_CUDA(cudaMalloc(&blurInputs, 128 * 128 * maxBlurBatchSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&blurOutputs, maxBlurBatchSize * sizeof(float)));

    blurCtx->setInputTensorAddress("input", blurInputs);
    blurCtx->setOutputTensorAddress("output", blurOutputs);

    //std::thread([this] {process_blur_Q();}).detach();
  }

  void StreamCam::launch_blur_metric() {
    blurMutex.lock();
    if (blurMeticQ.empty()) {
      blurMutex.unlock();
      return;
    }
    ++iters;

    blurImagesInProcess.clear();

    while (!blurMeticQ.empty() && blurImagesInProcess.size() < maxBlurBatchSize - 1) {
      blurImagesInProcess.push_back(blurMeticQ.front());
      blurMeticQ.pop();
    }
    blurMutex.unlock();
    frames+=blurImagesInProcess.size();

    const size_t step = 128 * 128 * sizeof(float);
    for (int i = 0; i < blurImagesInProcess.size(); ++i) {
      assert(!blurImagesInProcess[i]->blurDFT.empty());

      auto dst = blurInputs + step * i;

      CHECK_CUDA(cudaMemcpy2DAsync(dst,
        128 * sizeof(float),
        blurImagesInProcess[i]->blurDFT.data,
        blurImagesInProcess[i]->blurDFT.step,
        128 * sizeof(float),
        128,
        cudaMemcpyHostToDevice,
        blurStream));
      auto err = cudaGetLastError();
    }
    // //verification of image input
    // Mat test1;
    // cuda::GpuMat test(128, 128,CV_32FC1, blurInputs);
    // test.download(test1);

    Dims4 inDims{static_cast<int>(blurImagesInProcess.size()), 1, 128, 128};
    assert(blurCtx->setInputShape("input", inDims));

    assert(blurCtx->enqueueV3(blurStream));
    outstandingBlurInference = true;
    // for (int i = 0; i < blurImagesInProcess.size(); ++i) {
    //   if (blurImagesInProcess[i]->index == 0) {
    //     int k = 0;
    //   }
    // }
  }

  void StreamCam::receive_blur_metric() {
    if (!outstandingBlurInference){return;}
    // for (int i = 0; i < blurImagesInProcess.size(); ++i) {
    //   if (blurImagesInProcess[i]->index == 0) {
    //     int k = 0;
    //   }
    // }

    CHECK_CUDA(cudaStreamSynchronize(blurStream));

    float logit;
    for (int i = 0; i < blurImagesInProcess.size(); ++i) {
      cudaMemcpy(&logit, blurOutputs + i, sizeof(float), cudaMemcpyDeviceToHost);
      blurImagesInProcess[i]->motionBlur = 1.f / (1.f + std::exp(-logit));

      std::lock_guard lock(blurImagesInProcess[i]->blurMutex);
      blurImagesInProcess[i]->blurSet = true;
      blurImagesInProcess[i]->blurConVar.notify_one();
    }
    outstandingBlurInference = false;
  }
}
