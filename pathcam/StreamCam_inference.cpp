//
// Created by Cooper Maira on 12/10/25.
//

#include "pathCam.h"

int maxBatchSize = 64;

namespace pathCam {
  using namespace nvinfer1;

  void StreamCam::load_blur_engine() {
    auto dBlob = readFile(no_ref_blur_model_path.toString());

    IRuntime *bRuntime = createInferRuntime(nvloger);
    blurEngine = bRuntime->deserializeCudaEngine(dBlob.data(), dBlob.size());
    delete bRuntime;

    assert(blurEngine);
    blurCtx = blurEngine->createExecutionContext();
    cudaStreamCreate(&blurStream);

    CHECK_CUDA(cudaMalloc(&blurInputs,128 * 128 * maxBlurBatchSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&blurOutputs,maxBlurBatchSize * sizeof(float)));
    // blurInputs = static_cast<char *>(malloc(128 * 128 * maxBlurBatchSize * sizeof(float)));
    // blurOutputs = static_cast<float *>(malloc(maxBlurBatchSize * sizeof(float)));

    blurCtx->setInputTensorAddress("input", blurInputs);
    blurCtx->setOutputTensorAddress("output", blurOutputs);
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

  void StreamCam::launch_blur_metric() {
    blurMutex.lock();
    if (blurMeticQ.empty()) {
      blurMutex.unlock();
      return;
    }

    std::vector<Image *> imgs;
    while (!blurMeticQ.empty() && imgs.size() < maxBlurBatchSize - 1) {
      imgs.push_back(blurMeticQ.front());
      blurMeticQ.pop();
    }
    blurMutex.unlock();

    const size_t step = 128 * 128 * sizeof(float);
    for (int i = 0; i < imgs.size(); ++i) {
      assert(!imgs[i]->blurDFT.empty());

      auto dst = blurInputs + step * i;

      CHECK_CUDA(cudaMemcpy2DAsync(dst,
        128*sizeof(float),
        imgs[i]->blurDFT.data,
        imgs[i]->blurDFT.step,
        128 * sizeof(float),
        128,
        cudaMemcpyHostToDevice,
        blurStream));
      auto err = cudaGetLastError();
      // memcpy(dst,imgs[i]->blurDFT.data,step);
    }
    // Mat test1;
    Mat test(128, 128,CV_32FC1, blurInputs);
    // test.download(test1);

    Dims4 inDims{static_cast<int>(imgs.size()), 1, 128, 128};
    assert(blurCtx->setInputShape("input", inDims));

    assert(blurCtx->enqueueV3(blurStream));

    CHECK_CUDA(cudaStreamSynchronize(blurStream));


    for (int i = 0; i < imgs.size(); ++i) {
      // imgs[i]->motionBlur = blurOutputs[i];
      cudaMemcpy(&imgs[i]->motionBlur, &blurOutputs[i], sizeof(float), cudaMemcpyDeviceToHost);
    }


    int k = 0;
  }
}
