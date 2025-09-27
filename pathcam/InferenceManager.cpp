//
//  InferenceManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 1/3/25.
//

/*As described here https://github.com/pytorch/pytorch/issues/75932 libtorch can deallocate tensors when working with embedded pytorch code
 *even when strong references (ie member variable refs) still exist in the c++ code. This may be fixed with a new version of torch, as this code
 *was written for 2.0.1. For now, a workaround is to never create tensor member variables. Declare them locally only, transforming them to vector
 *or pylist etc before a function ends if they need to persist.
 *
 *Also, I have been unable to acquire the pyGIL from any other thread then where py_Initialize is called. This is also pretty finicky as you can
 *usually still step into embedded python, but cannot send things to the GPU once there if you haven't acquired the GIL in your calling c++.
 */

//#include <bits/fs_fwd.h>

#include "pathCam.h"

namespace pathCam {
  InferenceManager::InferenceManager(StreamCam *parent) : parent(parent), device(torch::Device(torch::kCPU)),
                                                          tileEmbedMutex(Poco::FastMutex()) {


    embedFileOut = parent->slide_encoder_path.toString() + "/tileEmbeds.pt";
    coordsFileOut = parent->slide_encoder_path.toString() + "/coords.pt";
    signalFileOut = parent->slide_encoder_path.toString() + "/pSignal.txt";

    reportFileIn = parent->slide_encoder_path.toString() + "/response.txt";
    embedFileIn = parent->slide_encoder_path.toString() + "/response.pt";
    signalFileIn = parent->slide_encoder_path.toString() + "/cSignal.txt";



  }


//   Inferencer::Inferencer(StreamCam *_parent) : PostProcessorBase(_parent) {
//     /*
//     embedFileOut = parent->slide_encoder_path.toString() + "/tileEmbeds.pt";
//     coordsFileOut = parent->slide_encoder_path.toString() + "/coords.pt";
//     signalFileOut = parent->slide_encoder_path.toString() + "/pSignal.txt";
//
//     reportFileIn = parent->slide_encoder_path.toString() + "/response.txt";
//     embedFileIn = parent->slide_encoder_path.toString() + "/response.pt";
//     signalFileIn = parent->slide_encoder_path.toString() + "/cSignal.txt";
// */
//
//
//   }


  void InferenceManager::run() {
    auto startTime = std::chrono::high_resolution_clock::now();

    int inferenceDevice = -1;

    torch::NoGradGuard noGrad;

    size_t dstPitch;
    size_t elementSize;
    size_t tileSizeInBytes;

    if (torch::cuda::is_available()) {
      inferenceDevice = parent->GPU_select_cuda_device(0);
      device = torch::Device(torch::kCUDA, inferenceDevice);

#ifdef HAVE_OPENCV_CUDAARITHM

      if (inferenceDevice != parent->compositorCudaDevice) {
        //create buffers for moving data from gpu1 to gpu2
        size_t bufferSize = parent->tileSize * parent->tileSize * parent->maxTilesPerBatch;
        bufferMemory = static_cast<char *>(malloc(bufferSize * 4));

        //set cuda memory on inference gpu
        CHECK_CUDA(cudaSetDevice(inferenceDevice));
        CHECK_CUDA(cudaMalloc(&bufferGPU,bufferSize * 4));
        CHECK_CUDA(cudaMalloc(&bufferGPU_rcv,bufferSize * 3));

        elementSize = 4;
        dstPitch = elementSize * parent->tileSize;
        tileSizeInBytes = 4 * parent->tileSize * parent->tileSize;
      }

#endif

      std::cout << "Torch using GPU (CUDA) with device: " << device << std::endl;
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


    {
      // Load the tile encoder TorchScript model and move it to the selected device
      auto val = parent->tile_encoder_path.toString();
      //auto tileEncoderModel = torch::jit::load(val);
      //tileEncoderModel.to(device);
      torch::jit::Module tileEncoderModel = torch::jit::load(val, c10::Device(torch::kCUDA, inferenceDevice));
      tileEncoderModel.eval();


      //set standard deviation and mean tensors to match imageNet normalization
      auto mean = torch::tensor({0.485, 0.456, 0.406}, torch::kFloat32).view({1, 3, 1, 1}).to(device);
      auto stddv = torch::tensor({0.229, 0.224, 0.225}, torch::kFloat32).view({1, 3, 1, 1}).to(device);

      at::Tensor batch_tensor;
      std::cout<<"model loaded "<<(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime)).count()<<std::endl;

      while (parent->compositing || !parent->tileEmbedQ.empty()) {
        auto tileList = parent->get_tile_embed_Q_front();

        if (tileList.empty()) {
          parent->inferenceWait.wait();
        } else {
          size_t numImages = tileList.size();

          //keep track of component membership
          unsigned int component = std::get<2>(tileList[0]);
          if (minShift.size() < component + 1) {
            minShift.push_back({0, 0});
          }

          //get reference to pyramid tiles
          auto pyramidLevel = parent->composites[component]->imagePyramid->level[0];

          // auto batch_tensor = torch::empty({static_cast<int64_t>(numImages), parent->tileSize, parent->tileSize, 3},
          //                                  torch::kFloat32).to(device);

          tileEmbedMutex.lock();
          cuda::setDevice(parent->compositorCudaDevice);

          for (int i = 0; i < numImages; i++) {

            //add to coord dict
            if (tileCoordToTensorIndex.find(tileList[i]) == tileCoordToTensorIndex.end()) {
              tileCoordToTensorIndex.insert({tileList[i], tileCoordToTensorIndex.size()});
            }

            //adjust mins (later used for adjustment)
            if (std::get<0>(tileList[i]) < minShift[component].first) {
              minShift[component].first = std::get<0>(tileList[i]);
            }
            if (std::get<1>(tileList[i]) < minShift[component].second) {
              minShift[component].second = std::get<1>(tileList[i]);
            }

#ifdef HAVE_OPENCV_CUDAARITHM

            int x = std::get<0>(tileList[i]);
            int y = std::get<1>(tileList[i]);
            threeChannelPrealGPU = pyramidLevel->getTile(x, y).image;
            uchar* bufferPtr = reinterpret_cast<uchar*>(bufferMemory + i * tileSizeInBytes);
            CHECK_CUDA(cudaMemcpy2D(bufferPtr,dstPitch,threeChannelPrealGPU.data,threeChannelPrealGPU.step,dstPitch,threeChannelPrealGPU.rows,cudaMemcpyDeviceToHost));

#else
            cvtColor(pyramidLevel->getTile(std::get<0>(tileList[i]), std::get<1>(tileList[i])),
                     threeChannelPreallocated,
                     COLOR_BGRA2RGB);

            //clone would be necessary if not immediately moved to gpu. Tensor from_blob keeps ref to orig obj
            batch_tensor[i] = torch::from_blob(threeChannelPreallocated.data, {parent->tileSize, parent->tileSize, 3},
                                               torch::kUInt8).to(device);
#endif
          }

          //get data to inferencing GPU
          cudaSetDevice(inferenceDevice);
          cudaMemcpy(bufferGPU,bufferMemory,numImages * tileSizeInBytes,cudaMemcpyHostToDevice);

          //equivalent of cvtColor(...,COLOR_BGRA2RGB) but done as a single kernel call
          int numPixels = numImages * parent->tileSize * parent->tileSize;
          launch_drop_alpha_and_swap(bufferGPU_rcv, bufferGPU, numPixels);
          cudaDeviceSynchronize();

          batch_tensor = torch::from_blob(bufferGPU_rcv, {(long) numImages, parent->tileSize, parent->tileSize, 3},
            torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, inferenceDevice));

          //permute to N,C,W,H, center crop, make memory block index-contiguous via type conversion,
          //normalize to [0,1], normalize to imageNet mean/stddv
          batch_tensor = batch_tensor.permute({0, 3, 1, 2})
                  //.slice(2, cropLoc, cropLoc + cropedDim)
                  //.slice(3, cropLoc, cropLoc + cropedDim)
              .to(torch::kFloat32)
              .div(255)
              .sub(mean)
              .div(stddv);

          //run inference on batch of images
          auto output = tileEncoderModel.forward({batch_tensor}).toTensor();

          if (tileEmbeds.sizes().size() == 1) {
            tileEmbeds = output;
            std::cout << output.sizes() << std::endl;
          } else {
            //extend tensor to match coord dict
            int extendBy = tileCoordToTensorIndex.size() - tileEmbeds.sizes()[0];
            tileEmbeds = torch::cat({
                                        tileEmbeds, torch::empty({extendBy, output.sizes()[1]},
                                                                 torch::TensorOptions().dtype(torch::kFloat32).device(
                                                                     device))
                                    });

            //add new tile embedding to the list, if a tile has been run previously, overwrite it
            for (int i = 0; i < numImages; i++) {
              auto locationInList = tileCoordToTensorIndex[tileList[i]];
              tileEmbeds[locationInList] = output[i];
            }
          }

          tileEmbedMutex.unlock();
          std::cout << "Processed " + std::to_string(tileList.size()) << std::endl;
        }

      }

    }//scope to expire 4+ gb tile encoder model
    tileEmbeds = tileEmbeds.to(torch::kCPU);

    std::cout << "Tile Embedding Completed in " <<(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime)).count()<< std::endl;
    parent->tileEmbeddingComplete = true;


    if (parent->reportText) {
      try {
        std::ifstream file(reportFileIn);

        std::stringstream buffer;
        buffer << file.rdbuf();

        //reset dir
        remove(signalFileIn.c_str());
        remove(reportFileIn.c_str());

        std::cout << buffer.str() << std::endl;

      } catch (const Poco::Exception &exc) {
        std::cerr << "Error reading file: " << exc.displayText() << std::endl;
      }

    }
    //TODO remove this. this just prevents runnableEntry from calling the im destructor
    //while (true){}
  }

  void InferenceManager::run_agg_classify() {
    torch::NoGradGuard noGrad;
    Poco::ScopedLock<Poco::FastMutex> lock(tileEmbedMutex);
    torch::Tensor embedsToClassify;

    //create coords tensor for handoff
    torch::Tensor coordsTensor = torch::empty({long(tileCoordToTensorIndex.size()), 3},
                                              torch::TensorOptions().dtype(torch::kFloat32));
    for (const auto &[key, value]: tileCoordToTensorIndex) {
      coordsTensor[value][0] = std::get<2>(key);        //component index
      coordsTensor[value][1] = std::get<0>(key) - minShift[std::get<2>(key)].first; //tile x coord
      coordsTensor[value][2] = std::get<1>(key) - minShift[std::get<2>(key)].second; //tile y coord
    }

    if (parent->aggregatingHandoff) {
      //file exchange

      //pack as disk savable tensors
      //tileEmbeds = tileEmbeds.to(torch::kCPU);
      auto pickledEmbed = torch::pickle_save(tileEmbeds);
      std::ofstream feout = std::ofstream(embedFileOut);
      feout.write(pickledEmbed.data(), pickledEmbed.size());
      feout.close();
      tileEmbeds = tileEmbeds.to(device);

      auto pickledCoords = torch::pickle_save(coordsTensor);
      std::ofstream fcout = std::ofstream(coordsFileOut);
      fcout.write(pickledCoords.data(), pickledCoords.size());
      fcout.close();

      //create signal file
      std::ofstream fsout = std::ofstream(signalFileOut);
      fsout.close();

      //wait for response
      while (!std::ifstream(signalFileIn).good()) {
        continue;
      }
      //load response, embeds are at index specified by tileCoordToTensorIndex
      std::ifstream fin(embedFileIn, std::ios::binary);
      std::vector<char> buffer((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
      torch::Tensor aggregatedEmbeds = torch::pickle_load(buffer).toTensor();
      //update tileEmbeds after attention
      embedsToClassify = torch::cat({aggregatedEmbeds, coordsTensor}, 1).to(device).to(torch::kF16);

      //reset dir
      remove(signalFileIn.c_str());
      remove(embedFileIn.c_str());
    } else {
      coordsTensor = coordsTensor.to(device);
      embedsToClassify = torch::cat({tileEmbeds, coordsTensor}, 1).to(torch::kF32);
    }


    //setup classifier model
    auto val = parent->classifier_path.toString();
    auto classifier = torch::jit::load(val);
    classifier.eval();
    classifier.to(device);


    //run and argmax
    auto classes = classifier.forward({embedsToClassify}).toTensor();
    classes = std::get<1>(classes.max(1));
    classes = classes.to(torch::kCPU);

    //build tile to class dict
    for (auto [key, value]: tileCoordToTensorIndex) {
      parent->tileCoordToClass[key] = classes[value].item<int>();
    }

    parent->update_observers();

  }
}
