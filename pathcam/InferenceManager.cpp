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

#include "pathCam.h"

namespace pathCam {
    InferenceManager::InferenceManager(StreamCam *parent) : parent(parent), device(torch::kCPU),
                                                            aggregatorMutex(new Poco::FastMutex()),
                                                            aggregatorWait(false),
                                                            adapter(*this, &InferenceManager::initialize_aggregator) {
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

        //load slide aggregator
        if (parent->slideEncoding) {
#ifdef WITH_MPS
            //on its own thread doesnt work on apple
            //https://github.com/python/cpython/issues/123022
            initializeModel();
#else
            auto initRunnable = new InferenceInitRunner(this);
            thread.start(initRunnable);

#endif
            //initialize_aggregator();
        }
        // Load the tile encoder TorchScript model and move it to the selected device
        auto val = parent->tile_encoder_path.toString();
        tileEncoderModel = torch::jit::load(val);
        tileEncoderModel.eval();
        tileEncoderModel.to(device);

        //coord shifting values
        minx = 0;
        miny = 0;

        //set crop dimension, this should probably be configurable;
        cropedDim = 224;

        //set size of embed vector, again this should be configurable
        embedSize = 1536;

        tileCoordToTensorIndex = new std::map<Point2i,unsigned long,PointComparator>();
    }

    void InferenceManager::run() {
        torch::NoGradGuard noGrad;

        
        //empty tileEmbeds, just needs to be initialized
        auto tileEmbeds = torch::empty({0, embedSize}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        //set standard deviation and mean tensors to match imageNet normalization
        auto mean = torch::tensor({0.485, 0.456, 0.406}, torch::kFloat32).view({1, 3, 1, 1}).to(device);
        auto stddv = torch::tensor({0.229, 0.224, 0.225}, torch::kFloat32).view({1, 3, 1, 1}).to(device);

        
        while (parent->compositing) {
            auto tileList = parent->get_tile_embed_Q_front();

            if (tileList.empty()) {
                //parent->inferenceWait.wait();
                continue;
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
                    if (tileCoordToTensorIndex->find(tileList[i].first) == tileCoordToTensorIndex->end()) {
                        tileCoordToTensorIndex->insert({tileList[i].first, tileCoordToTensorIndex->size()});
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
                        .slice(2, cropLoc, cropLoc + cropedDim)
                        .slice(3, cropLoc, cropLoc + cropedDim)
                        .to(torch::kFloat32)
                        .div(255)
                        .sub(mean)
                        .div(stddv);


                //extend tensor to match coord dict
                int extendBy = tileCoordToTensorIndex->size() - tileEmbeds.sizes()[0];
                tileEmbeds = torch::cat({
                    tileEmbeds, torch::empty({extendBy, embedSize},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(
                                                 device))
                });

                //run inference on batch of images
                auto output = tileEncoderModel.forward({batch_tensor}).toTensor();

                //add new tile embedding to the list, if a tile has been run previously, overwrite it
                for (int i = 0; i < numImages; i++) {
                    auto locationInList = (*tileCoordToTensorIndex)[tileList[i].first];
                    tileEmbeds[locationInList] = output[i];
                }
            }
            std::cout<<"Processed "+std::to_string(tileList.size())<<std::endl;
        }
        tileEmbeds = tileEmbeds.to(torch::kCPU);


        auto coordsTensor = torch::empty({tileCoordToTensorIndex->size(), 2}, torch::TensorOptions().dtype(torch::kFloat32));
        for (const auto &[key,value]: *tileCoordToTensorIndex) {
            std::cout << key << " " << value << "\n";
            coordsTensor[value][0] = key.x - minx;
            coordsTensor[value][1] = key.y - miny;
        }
        tileEmbedVec = tensorToVector(tileEmbeds);
        coordsVec = tensorToVector(coordsTensor);

        std::cout<<"Tile Embedding Complete"<<std::endl;

        aggregatorMutex->lock();
        aggregatorReady = true;
        aggregatorWait.set();
        aggregatorMutex->unlock();

        tileEncoderModel = torch::jit::Module();
        parent->tileEmbeddingComplete = true;
        ///run_slide_aggregation();
    }

    void InferenceManager::run_slide_aggregation() {
        //make sure aggregator initialization is complete
        aggregatorMutex->lock();
        if (!aggregatorReady) {
            aggregatorMutex->unlock();
            aggregatorWait.wait();
        }else {
            aggregatorMutex->unlock();
        }

        //make tensor of coords where coords are kept at proper index

        PyGILState_STATE gstate = PyGILState_Ensure();
        // Import the Python function
        //PyObject *pFuncForwardStep = PyObject_GetAttrString(pModule, "forward_step");
        PyObject *pFuncForwardStep = PyObject_GetAttrString(pModule, "forward_step_from_lists");
        if (!pFuncForwardStep || !PyCallable_Check(pFuncForwardStep)) {
            PyErr_Print();
            throw std::runtime_error("Failed to locate 'forward_step' function.");
        }

        //package it up for the python call
        //auto pyCoords = py::reinterpret_steal<py::object>(THPVariable_Wrap(coordsTensor));
        //auto pyTileEmbeds = py::reinterpret_steal<py::object>(THPVariable_Wrap(tileEmbeds));

        auto pyCoords = tensorToList2(vectorToTensor(coordsVec));
        auto pyTileEmbeds = tensorToList2(vectorToTensor(tileEmbedVec));
        auto pyArgs = PyTuple_Pack(3,slideAggregator,pyTileEmbeds,pyCoords);


        std::cout<<"Calling Aggregator"<<std::endl;
        PyObject *pyResult = PyObject_Call(pFuncForwardStep, pyArgs, NULL);
        auto attendedTileEmbeds = THPVariable_Unpack(pyResult);
        int k = 0;
        PyGILState_Release(gstate);

        // Py_XDECREF(pArgs);
        // Py_XDECREF(pTileEmbeds);
        // Py_XDECREF(pCoords);
        // Py_XDECREF(pFuncForwardStep);

        // if (pOutput) {
        //     // Process the output (example)
        //     PyObject *pClsToken = PyTuple_GetItem(pOutput, 0);
        //     PyObject *pAttentionEmbeds = PyTuple_GetItem(pOutput, 1);
        //
        //     // Print results (or convert to C++ objects as needed)
        //     std::cout << "Output received successfully." << std::endl;
        //
        //     Py_XDECREF(pClsToken);
        //     Py_XDECREF(pAttentionEmbeds);
        //     Py_XDECREF(pOutput);
        // } else {
        //     PyErr_Print();
        // }
    }

    void InferenceManager::initialize_aggregator() {

        std::string venv_python_path = parent->python_venv_path.toString(); // Update with your venv Python path

        // Path to the directory containing model_inference.py
        std::string model_script_dir = parent->slide_encoder_path.parent().toString(); // Update with your path
        std::string python_file_name = parent->slide_encoder_path.getBaseName();
        // Python module name (no .py extension)

        std::string venvPath = "/media/max/pathCAM/gigapath-venv-linux";
        //std::string venvPath = "/home/max/venv-3";
        std::string oldPath = std::getenv("PATH") ? std::getenv("PATH") : "";

        // 1) Ensure the venv bin is first in PATH:
        std::string newPath = venvPath + "/bin:" + oldPath;
        setenv("PATH", newPath.c_str(), 1);

        // 2) Virtual env pointer
        setenv("VIRTUAL_ENV", venvPath.c_str(), 1);

        // 4) Let dynamic loader find PyTorch libs:
        std::string oldLD = std::getenv("LD_LIBRARY_PATH") ? std::getenv("LD_LIBRARY_PATH") : "";
        std::string newLD = venvPath + "/lib/python3.10/site-packages/torch/lib:" + oldLD;
        setenv("LD_LIBRARY_PATH", newLD.c_str(), 1);

        // Now initialize Python
        Py_Initialize();
        //PyEval_InitThreads();
        PyGILState_STATE gstate = PyGILState_Ensure();


        // Add the Python script directory to sys.path
        std::string add_path_command = "import sys; sys.path.append('" + model_script_dir + "')";
        PyRun_SimpleString(add_path_command.c_str());

        // Import the Python module
        pModule = PyImport_ImportModule(python_file_name.c_str());


        if (!pModule) {
            PyErr_Print();
            throw std::runtime_error("Failed to import Python module.");
        }

        // Call load_model to retrieve the model
        PyObject *pFuncLoadModel = PyObject_GetAttrString(pModule, "load_model");
        if (!pFuncLoadModel || !PyCallable_Check(pFuncLoadModel)) {
            PyErr_Print();
            throw std::runtime_error("Failed to locate 'load_model' function.");
        }

        PyObject *pModel = PyObject_CallObject(pFuncLoadModel, nullptr);
        Py_XDECREF(pFuncLoadModel);
        Py_XDECREF(pModule);

        if (!pModel) {
            PyErr_Print();
            throw std::runtime_error("Failed to load the model.");
        }

        slideAggregator = pModel;
        PyGILState_Release(gstate);

        run_slide_aggregation();
    }


    PyObject* InferenceManager::tensorToList2(const torch::Tensor& tensor) {
            if (!tensor.device().is_cpu()) {
        throw std::runtime_error("Tensor must be on CPU before conversion to a Python list.");
    }

    // Ensure the tensor is contiguous for efficient access
    torch::Tensor contiguous_tensor = tensor.contiguous();

    // Create a Python list
    PyObject* py_list = PyList_New(0);

    // Handle 1D and multidimensional tensors
    if (contiguous_tensor.dim() == 1) {
        // Create a separate copy of the tensor data
        auto data_ptr = contiguous_tensor.data_ptr<float>();
        for (int64_t i = 0; i < contiguous_tensor.size(0); ++i) {
            // Copy the value into the Python list
            PyObject* py_value = PyFloat_FromDouble(static_cast<double>(data_ptr[i]));
            PyList_Append(py_list, py_value);
            Py_DECREF(py_value);  // PyList_Append increments the reference count
        }
    } else {
        // For multidimensional tensors, recursively handle sub-tensors
        for (int64_t i = 0; i < contiguous_tensor.size(0); ++i) {
            // Create a separate sub-list for each dimension
            PyObject* sub_list = tensorToList2(contiguous_tensor[i]);
            PyList_Append(py_list, sub_list);
            Py_DECREF(sub_list);
        }
    }

    return py_list;
    }

    std::vector<std::vector<float>> InferenceManager::tensorToVector(const torch::Tensor& tensor) {
    // Ensure the tensor is contiguous
    torch::Tensor contigTensor = tensor.contiguous();
    auto sizes = tensor.sizes();

    // Validate dimensions
    if (sizes.size() != 2) {
        throw std::runtime_error("Tensor must be 2D to convert to vector of vectors.");
    }

    // Access tensor data
    const float* data = contigTensor.data_ptr<float>();
    size_t rows = sizes[0];
    size_t cols = sizes[1];

    // Create std::vector<std::vector<float>>
    std::vector<std::vector<float>> vec(rows, std::vector<float>(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            vec[i][j] = data[i * cols + j];
        }
    }

    return vec;
}

// Convert std::vector<std::vector<float>> to torch::Tensor
torch::Tensor InferenceManager::vectorToTensor(const std::vector<std::vector<float>>& vec) {
    size_t rows = vec.size();
    size_t cols = vec[0].size();

    // Flatten std::vector<std::vector<float>> to 1D std::vector<float>
    std::vector<float> flatVec;
    flatVec.reserve(rows * cols);
    for (const auto& row : vec) {
        flatVec.insert(flatVec.end(), row.begin(), row.end());
    }

    // Create a torch::Tensor from the flat vector and reshape
    return torch::from_blob(flatVec.data(), {static_cast<int64_t>(rows), static_cast<int64_t>(cols)}).clone();
}

    void InferenceManager::run_slide_analysis() {
    }
}
