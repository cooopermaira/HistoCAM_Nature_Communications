//
//  InferenceManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 1/3/25.
//

#include "pathCam.h"

namespace pathCam {

InferenceManager::InferenceManager(StreamCam* parent):parent(parent){
  // Check for available devices and set the device accordingly
  torch::Device device(torch::kCPU);
  
  if (torch::cuda::is_available()) {
    std::cout << "Using GPU (CUDA)." << std::endl;
    device = torch::Device(torch::kCUDA);
  } else {
    std::cout << "CUDA not available. Using CPU." << std::endl;
  }
  
  // Load the TorchScript model and move it to the selected device
  auto val = parent->tile_encoder_path.toString();
  torch::jit::script::Module model =  torch::jit::load(val);
  model.to(device);
  
}

void InferenceManager::run(){}

}
