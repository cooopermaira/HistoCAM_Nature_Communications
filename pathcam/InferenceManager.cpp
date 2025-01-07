//
//  InferenceManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 1/3/25.
//

#include "pathCam.h"

namespace pathCam {



  InferenceManager::InferenceManager(StreamCam *parent) : parent(parent), device(torch::kCPU) {
    if (torch::cuda::is_available()) {
      std::cout << "Using GPU (CUDA)" << std::endl;
      device = torch::Device(torch::kCUDA);
    }
#ifdef WITH_MPS
    else if (torch::mps::is_available()) {
      std::cout << "Using GPU (MPS)" << std::endl;
      device = torch::Device(torch::kMPS);
    }
#endif
    else {
      std::cout << "GPU not available. Using CPU." << std::endl;
    }

    // Load the TorchScript model and move it to the selected device
    auto val = parent->tile_encoder_path.toString();
    model = torch::jit::load(val);
    model.eval();
    model.to(device);



    //set crop dimension, this should probably be configurable;
    cropedDim = 224;

    //set size of embed vector, again this should be configurable
    embedSize = 1536;

    tileEmbeds = torch::empty({0,embedSize},torch::TensorOptions().dtype(torch::kFloat32).device(device));

    //set standard deviation and mean tensors to match imageNet normalization
    mean = torch::tensor({0.485,0.456,0.406},torch::kFloat32).view({1,3,1,1}).to(device);
    stddv = torch::tensor({0.229,0.224,0.225},torch::kFloat32).view({1,3,1,1}).to(device);
    //tileEmbeds = torch::empty({1,1536})
  }

  void InferenceManager::run() {
    int count = 0;
    torch::NoGradGuard noGrad;
    while (parent->compositing) {

      auto tileList = parent->get_tile_embed_Q_front();

      if (tileList.empty()) {
        parent->inferenceWait.wait();
      } else {
        unsigned int component = tileList[0].second;
        size_t numImages = tileList.size();
        auto pyramidLevel = parent->composites[component]->imagePyramid->level[0];
        int tileSize = parent->composites[component]->imagePyramid->tile_size;
        auto batch_tensor = torch::empty({static_cast<int64_t>(numImages), tileSize, tileSize, 3},torch::kFloat32);
        int cropLoc = (tileSize - cropedDim) / 2;

        for (int i = 0; i < numImages; i++) {

          //add to coord dict
          if(tileCoordToTensorIndex.find(tileList[i].first) == tileCoordToTensorIndex.end()){
            tileCoordToTensorIndex.insert({tileList[i].first,tileCoordToTensorIndex.size()});
          }else{
            parent->duplicateInferenceCount++;
          }

          cvtColor(pyramidLevel->getTile(tileList[i].first.x, tileList[i].first.y), threeChannelPreallocated,
                   COLOR_BGRA2RGB);
          //clone would be necessary if not immediately moved to gpu
          batch_tensor[i] = torch::from_blob(threeChannelPreallocated.data, {tileSize, tileSize, 3});

          count++;
        }
        //send data before cropping. not sure if this is the way to go since you send data you end up cropping out.
        batch_tensor = batch_tensor.to(device);

        //center crop, make memory block index contiguous, normalize to [0,1], normalize to imageNet mean/stddv
        batch_tensor = batch_tensor.permute({0, 3, 1, 2})
            .slice(2, cropLoc, cropLoc + cropedDim)
            .slice(3, cropLoc, cropLoc + cropedDim)
            .contiguous()
            .div(255)
            .sub(mean)
            .div(stddv);



        //extend tensor to match coord dict
        int extendBy = tileCoordToTensorIndex.size() - tileEmbeds.sizes()[0];
        tileEmbeds = torch::cat({tileEmbeds, torch::empty({extendBy, embedSize},torch::TensorOptions().dtype(torch::kFloat32).device(device))});

        //run inference on batch of images
        auto output = model.forward({batch_tensor}).toTensor();

        //add new tile embedding to the list, if a tile has been run previously, overwrite it
        for (int i = 0; i < numImages; i++){
          auto locationInList = tileCoordToTensorIndex[tileList[i].first];
          tileEmbeds[locationInList] = output[i];
        }

      }
    }
    int k = 0;
  }

}
