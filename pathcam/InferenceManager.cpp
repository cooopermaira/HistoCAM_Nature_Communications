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
    InferenceManager::InferenceManager(StreamCam *parent) : parent(parent){


        //load slide aggregator
//         if (parent->aggregating) {
// #ifdef WITH_MPS
//             //on its own thread doesnt work on apple
//             //https://github.com/python/cpython/issues/123022
//             initializeModel();
// #else
//             auto initRunnable = new InferenceInitRunner(this);
//             thread.start(initRunnable);
//
// #endif
//             //initialize_aggregator();
//         }

        //coord shifting values
        minx = 0;
        miny = 0;

        //set crop dimension, this should probably be configurable;
        cropedDim = 224;

        //set size of embed vector, again this should be configurable
        embedSize = 1536;

        tileCoordToTensorIndex = std::map<Point2i,unsigned long,PointComparator>();
    }


    void InferenceManager::run() {
        torch::NoGradGuard noGrad;
        auto device = torch::Device(torch::kCPU);

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

        
        //empty tileEmbeds, just needs to be initialized
        torch::Tensor tileEmbeds = torch::empty({0, embedSize}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        {
            // Load the tile encoder TorchScript model and move it to the selected device
            auto val = parent->tile_encoder_path.toString();
            auto tileEncoderModel = torch::jit::load(val);
            tileEncoderModel.to(device);

            //set standard deviation and mean tensors to match imageNet normalization
            auto mean = torch::tensor({0.485, 0.456, 0.406}, torch::kFloat32).view({1, 3, 1, 1}).to(device);
            auto stddv = torch::tensor({0.229, 0.224, 0.225}, torch::kFloat32).view({1, 3, 1, 1}).to(device);


            while (parent->compositing || !parent->tileEmbedQ.empty()) {
                auto tileList = parent->get_tile_embed_Q_front();

                if (tileList.empty()) {
                    parent->inferenceWait.wait();
                } else {
                    unsigned int component = tileList[0].second;
                    size_t numImages = tileList.size();
                    auto pyramidLevel = parent->composites[component]->imagePyramid->level[0];
                    int tileSize = parent->composites[component]->imagePyramid->tile_size;
                    auto batch_tensor = torch::empty({static_cast<int64_t>(numImages), tileSize, tileSize, 3},
                                                     torch::kFloat32).to(device);
                    int cropLoc = (tileSize - cropedDim) / 2;

                    for (int i = 0; i < numImages; i++) {
                        //add to coord dict
                        if (tileCoordToTensorIndex.find(tileList[i].first) == tileCoordToTensorIndex.end()) {
                            tileCoordToTensorIndex.insert({tileList[i].first, tileCoordToTensorIndex.size()});
                        }



                        //adjust mins (later used for adjustment)
                        if (tileList[i].first.x < minx) { minx = tileList[i].first.x; }
                        if (tileList[i].first.y < miny) { miny = tileList[i].first.y; }

                        cvtColor(pyramidLevel->getTile(tileList[i].first.x, tileList[i].first.y), threeChannelPreallocated,
                                 COLOR_BGRA2RGB);

                        //clone would be necessary if not immediately moved to gpu. Tensor from_blob keeps ref to orig obj
                        batch_tensor[i] = torch::from_blob(threeChannelPreallocated.data, {tileSize, tileSize, 3},
                                                           torch::kUInt8).to(device);
                    }

                    //center crop, make memory block index contiguous, normalize to [0,1], normalize to imageNet mean/stddv
                    batch_tensor = batch_tensor.permute({0, 3, 1, 2})
                            //.slice(2, cropLoc, cropLoc + cropedDim)
                            //.slice(3, cropLoc, cropLoc + cropedDim)
                            .to(torch::kFloat32)
                            .div(255)
                            .sub(mean)
                            .div(stddv);


                    //extend tensor to match coord dict
                    int extendBy = tileCoordToTensorIndex.size() - tileEmbeds.sizes()[0];
                    tileEmbeds = torch::cat({
                        tileEmbeds, torch::empty({extendBy, embedSize},
                                                 torch::TensorOptions().dtype(torch::kFloat32).device(
                                                     device))
                    });

                    //run inference on batch of images
                    auto output = tileEncoderModel.forward({batch_tensor}).toTensor();

                    //add new tile embedding to the list, if a tile has been run previously, overwrite it
                    for (int i = 0; i < numImages; i++) {
                        auto locationInList = (tileCoordToTensorIndex)[tileList[i].first];
                        tileEmbeds[locationInList] = output[i];
                    }
                    std::cout<<"Processed "+std::to_string(tileList.size())<<std::endl;
                }

            }

        }//scope to expire 4+ gb tile encoder model
        tileEmbeds = tileEmbeds.to(torch::kCPU);

        std::cout<<"Tile Embedding Complete"<<std::endl;
        parent->tileEmbeddingComplete = true;

        if (parent->aggregating) {
            //file exchange
            std::string embedFileOut("/media/max/Data/2_20/Torch/handoff/tileEmbeds.pt");
            std::string coordsFileOut("/media/max/Data/2_20/Torch/handoff/coords.pt");
            std::string signalFileOut("/media/max/Data/2_20/Torch/handoff/pSignal.txt");

            std::string embedFileIn("/media/max/Data/2_20/Torch/handoff/response.pt");
            std::string signalFileIn("/media/max/Data/2_20/Torch/handoff/cSignal.txt");


            //create coords tensor for handoff
            torch::Tensor coordsTensor = torch::empty({long(tileCoordToTensorIndex.size()), 2}, torch::TensorOptions().dtype(torch::kFloat32));
            for (const auto &[key,value]: tileCoordToTensorIndex) {
                coordsTensor[value][0] = key.x - minx;
                coordsTensor[value][1] = key.y - miny;
            }

            //pack as disk savable tensors
            auto pickledEmbed = torch::pickle_save(tileEmbeds);
            std::ofstream feout = std::ofstream(embedFileOut);
            feout.write(pickledEmbed.data(), pickledEmbed.size());
            feout.close();

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
            std::ifstream fin(embedFileIn,std::ios::binary);
            std::vector<char> buffer((std::istreambuf_iterator<char>(fin)),std::istreambuf_iterator<char>());
            torch::Tensor aggregatedEmbeds = torch::pickle_load(buffer).toTensor();

            //reset signal
            remove(signalFileIn.c_str());

            //update tileEmbeds after attention
            tileEmbeds = torch::cat({aggregatedEmbeds,coordsTensor},1).to(torch::kF32);
            //tileEmbeds = aggregatedEmbeds;
        }

        if (parent->classifying) {

            //setup classifier model
            auto val = parent->classifier_path.toString();
            auto classifier = torch::jit::load(val);
            classifier.eval();
            classifier.to(device);
            tileEmbeds = tileEmbeds.to(device);


            //run and argmax
            auto classes = classifier.forward({tileEmbeds}).toTensor();
            classes = std::get<1>(classes.max(1));
            classes = classes.to(torch::kCPU);

            //build tile to class dict
            for (auto [key,value] : tileCoordToTensorIndex) {
                parent->tileCoordToClass[key] = classes[value].item<int>();
                //parent->tileCoordToClass[key];
                //debug
                 // if (classes[value].item<int>() != 0) {
                 //     std::cout<<key.x + minx<<","<<key.y+miny<<std::endl;
                 // }

                //debug build fake dict for testing
                // int keyhash = key.x - minx + key.y - miny;
                // if (keyhash == 0) {
                //     int k = 0;
                // }
                //
                // parent->tileCoordToClass[key] = (keyhash % 4);
            }
            parent->classifyingComplete = true;
            parent->update_observers();
        }
    }

}
