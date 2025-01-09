//
//  InferenceManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 1/3/25.
//

#include "pathCam.h"

namespace pathCam {


  InferenceManager::InferenceManager(StreamCam *parent) : parent(parent), device(torch::kCPU), adapter(*this, &InferenceManager::initializeModel) {
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

    //load slide encoder on its own thread
//    if(parent->slideEncoding) {
//      thread.start(adapter);
//    }
    initializeModel();
    // Load the tile encoder TorchScript model and move it to the selected device
    auto val = parent->tile_encoder_path.toString();
    tileEncoderModel = torch::jit::load(val);
    tileEncoderModel.eval();
    tileEncoderModel.to(device);



    //set crop dimension, this should probably be configurable;
    cropedDim = 224;

    //set size of embed vector, again this should be configurable
    embedSize = 1536;

    tileEmbeds = torch::empty({0, embedSize}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

    //set standard deviation and mean tensors to match imageNet normalization
    mean = torch::tensor({0.485, 0.456, 0.406}, torch::kFloat32).view({1, 3, 1, 1}).to(device);
    stddv = torch::tensor({0.229, 0.224, 0.225}, torch::kFloat32).view({1, 3, 1, 1}).to(device);
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
        auto batch_tensor = torch::empty({static_cast<int64_t>(numImages), tileSize, tileSize, 3}, torch::kFloat32);
        int cropLoc = (tileSize - cropedDim) / 2;

        for (int i = 0; i < numImages; i++) {

          //add to coord dict
          if (tileCoordToTensorIndex.find(tileList[i].first) == tileCoordToTensorIndex.end()) {
            tileCoordToTensorIndex.insert({tileList[i].first, tileCoordToTensorIndex.size()});
          } else {
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
        tileEmbeds = torch::cat({tileEmbeds, torch::empty({extendBy, embedSize},
                                                          torch::TensorOptions().dtype(torch::kFloat32).device(
                                                              device))});

        //run inference on batch of images
        auto output = tileEncoderModel.forward({batch_tensor}).toTensor();

        //add new tile embedding to the list, if a tile has been run previously, overwrite it
        for (int i = 0; i < numImages; i++) {
          auto locationInList = tileCoordToTensorIndex[tileList[i].first];
          tileEmbeds[locationInList] = output[i];
        }

      }
    }
    delete &tileEncoderModel;
    parent->tileEmbeddingComplete = true;
    run_slide_analysis();
  }

  void InferenceManager::initializeModel() {
    std::string venv_python_path = parent->python_venv_path.toString(); // Update with your venv Python path
    std::string errorMessage;

    // Path to the directory containing model_inference.py
    std::string model_script_dir = parent->slide_encoder_path.parent().toString(); // Update with your path
    std::string python_file_name = parent->slide_encoder_path.getBaseName();       // Python module name (no .py extension)


    // Initialize Python interpreter
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    wchar_t* program_name = Py_DecodeLocale(venv_python_path.c_str(), nullptr);
    if (!program_name) {
      throw std::runtime_error("Failed to decode venv path.");
    }

    PyStatus status = PyConfig_SetString(&config, &config.program_name, program_name);
    PyMem_RawFree(program_name);
    if (PyStatus_Exception(status)) {
      PyErr_Print();
      PyConfig_Clear(&config);
      throw std::runtime_error("Failed to set Python program name.");
    }

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
      PyErr_Print();
      PyConfig_Clear(&config);
      throw std::runtime_error("Failed to initialize Python interpreter.");
    }

    PyConfig_Clear(&config);

    // Add the Python script directory to sys.path
    std::string add_path_command = "import sys; sys.path.append('" + model_script_dir + "')";
    PyRun_SimpleString(add_path_command.c_str());

    // Import the Python module
    PyObject* pModule = PyImport_ImportModule(python_file_name.c_str());


    if (!pModule) {
      PyErr_Print();
      throw std::runtime_error("Failed to import Python module.");
    }

    // Call load_model to retrieve the model
    PyObject* pFuncLoadModel = PyObject_GetAttrString(pModule, "load_model");
    if (!pFuncLoadModel || !PyCallable_Check(pFuncLoadModel)) {
      PyErr_Print();
      throw std::runtime_error("Failed to locate 'load_model' function.");
    }

    PyObject* pModel = PyObject_CallObject(pFuncLoadModel, nullptr);
    Py_XDECREF(pFuncLoadModel);
    Py_XDECREF(pModule);

    if (!pModel) {
      PyErr_Print();
      throw std::runtime_error("Failed to load the model.");
    }

    slideEncoder = pModel;
  }

  void runForwardStep(PyObject* pModel, const std::vector<float>& tile_embeds, const std::vector<float>& coords) {
    // Import the Python module (already in sys.path)
    PyObject* pFuncForwardStep = PyObject_GetAttrString(PyImport_AddModule("__main__"), "forward_step");
    if (!pFuncForwardStep || !PyCallable_Check(pFuncForwardStep)) {
      PyErr_Print();
      throw std::runtime_error("Failed to locate 'forward_step' function.");
    }

    // Convert tile_embeds and coords to Python lists
    PyObject* pTileEmbeds = PyList_New(tile_embeds.size());
    PyObject* pCoords = PyList_New(coords.size());

    for (size_t i = 0; i < tile_embeds.size(); ++i) {
      PyList_SetItem(pTileEmbeds, i, PyFloat_FromDouble(tile_embeds[i]));
    }
    for (size_t i = 0; i < coords.size(); ++i) {
      PyList_SetItem(pCoords, i, PyFloat_FromDouble(coords[i]));
    }

    // Call the forward_step function
    PyObject* pArgs = PyTuple_Pack(3, pModel, pTileEmbeds, pCoords);
    PyObject* pOutput = PyObject_CallObject(pFuncForwardStep, pArgs);

    Py_XDECREF(pArgs);
    Py_XDECREF(pTileEmbeds);
    Py_XDECREF(pCoords);
    Py_XDECREF(pFuncForwardStep);

    if (pOutput) {
      // Process the output (example)
      PyObject* pClsToken = PyTuple_GetItem(pOutput, 0);
      PyObject* pAttentionEmbeds = PyTuple_GetItem(pOutput, 1);

      // Print results (or convert to C++ objects as needed)
      std::cout << "Output received successfully." << std::endl;

      Py_XDECREF(pClsToken);
      Py_XDECREF(pAttentionEmbeds);
      Py_XDECREF(pOutput);
    } else {
      PyErr_Print();
    }
  }


  void InferenceManager::run_slide_analysis() {
  }
}
