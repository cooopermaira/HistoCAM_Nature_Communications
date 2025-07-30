//
// Created by cooper maira on 7/25/25.
//

#include "AccessSAM.h"
#include "utils.hpp"


namespace pathCam {

  void ProcessImage(const std::string& encoder_path,
                  const std::string& decoder_path,
                  const std::string& img_path,
                  const std::string& bbox_file_path,
                  const std::string& output_jpg_path,
                  const std::string& precision,
                  const int decoder_batch_limit)
{
    // Get image and bbox filenames
    std::vector<std::string> image_names;

    for (const auto& entry : std::filesystem::directory_iterator(img_path))
    {
        image_names.push_back(entry.path().string());
    }

    // Create SAM2Image object
    std::unique_ptr<SAM2Image> sam2;
    cv::Size encoder_input_size(1024,
                                1024);  // Current encoder input size, affects decoder normalization
    sam2 = std::make_unique<SAM2Image>(
        encoder_path, decoder_path, encoder_input_size, precision, decoder_batch_limit);

    const size_t batch_size = 1;
    for (size_t i = 0; i < image_names.size(); i += batch_size)
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<cv::Mat> images_batch;
        std::vector<std::vector<cv::Rect>> box_coords_batch;
        // Calculate actual batch size for this iteration
        size_t current_batch_size = std::min(batch_size, image_names.size() - i);

        // Read images and bounding boxes
        for (size_t j = 0; j < current_batch_size; j++)
        {
            std::filesystem::path image_path = image_names[i + j];
            std::string image_file_name = image_path.filename().string();
            std::string bb_file_name;
            if (image_file_name.find(".jpg") != std::string::npos)
            {
                bb_file_name = ReplaceFileExtension(image_file_name, ".jpg", ".txt");
            }
            else if (image_file_name.find(".png") != std::string::npos)
            {
                bb_file_name = ReplaceFileExtension(image_file_name, ".png", ".txt");
            }

            // Read image and bounding box
            std::filesystem::path bb_file_path =
                std::filesystem::path(bbox_file_path) / bb_file_name;
            images_batch.push_back(cv::imread(image_path.string()));
            std::vector<cv::Rect> box_coords =
                ReadAndTransformCoordinates(bb_file_path.string());
            box_coords_batch.push_back(box_coords);
        }

        // Run encoder
        auto start_encoder = std::chrono::high_resolution_clock::now();
        sam2->RunEncoder(images_batch);
        auto end_encoder = std::chrono::high_resolution_clock::now();

        // Run decoder
        auto start_decoder = std::chrono::high_resolution_clock::now();
        sam2->RunDecoder(box_coords_batch);
        auto end_decoder = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<cv::Mat>> masks = sam2->GetMasks();

        auto start_draw = std::chrono::high_resolution_clock::now();
        for (size_t j = 0; j < current_batch_size; j++)
        {
          for (auto &mask : masks[j]) {
            imwrite(output_jpg_path + "_mask"+std::to_string(i+j)+".png",mask);
          }
            cv::Mat masked_img = DrawMasks(images_batch[j], masks[j]);
            cv::imwrite(output_jpg_path + "_" + std::to_string(i + j) + ".jpg", masked_img);
        }
        auto end_draw = std::chrono::high_resolution_clock::now();

        auto duration_encoder = std::chrono::duration<double>(end_encoder - start_encoder);
        std::cout << "Encoder time: " << duration_encoder.count() << "s" << std::endl;
        auto duration_decoder = std::chrono::duration<double>(end_decoder - start_decoder);
        std::cout << "Decoder time: " << duration_decoder.count() << "s" << std::endl;
        auto duration_draw = std::chrono::duration<double>(end_draw - start_draw);
        std::cout << "Draw time: " << duration_draw.count() << "s" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end - start);
        std::cout << "Total time(one iteration): " << duration.count() << "s" << std::endl;
    }
}
  SAMTile::SAMTile(int _ID, Point2i _location, unsigned int _size) : ID(_ID), location(_location), size(_size) {
    noncontiguousWrapper = cuda::GpuMat(size, size,CV_8UC4, Scalar(0, 0, 0, 0));
  }

  void SAMTile::set_component_tile(Point2i _tileID, Point2i _subLocation, cuda::GpuMat &_tileMat) {
    componentTiles.emplace_back(_subLocation, _tileID);
    Rect ROI(_subLocation.x * _tileMat.cols, _subLocation.y * _tileMat.rows, _tileMat.cols, _tileMat.rows);
    try {
      _tileMat.copyTo(noncontiguousWrapper(ROI));
    } catch (...) {
      int k = 0;
    }
  }

  void SAMTile::make_raw_buffer(void *_buffer) {
    //convert to F32, normalize for imagenet. keep in mind the mat is currently in BGRA
    noncontiguousWrapper.convertTo(noncontiguousWrapper,CV_32F);
    cuda::divide(noncontiguousWrapper, Scalar(255, 255, 255, 255), noncontiguousWrapper);
    cuda::subtract(noncontiguousWrapper, Scalar(0.406, 0.456, 0.485, 0), noncontiguousWrapper);
    cuda::divide(noncontiguousWrapper, Scalar(0.225, 0.224, 0.229, 1), noncontiguousWrapper);


    std::vector<cuda::GpuMat> split_channels;
    cuda::split(noncontiguousWrapper, split_channels);

    size_t nBytesPerChannel = sizeof(float) * noncontiguousWrapper.cols * noncontiguousWrapper.rows;
    //cudaMalloc(&rawBuffer, nBytesPerChannel * 3);

    for (int i = 0; i < 3; ++i) {
      void *dst = static_cast<char *>(_buffer) + i * nBytesPerChannel;
      CHECK_CUDA_ERROR(cudaMemcpy2D(dst,
        split_channels[2 - i].cols * sizeof(float),
        split_channels[2 - i].data,
        split_channels[2 - i].step,
        split_channels[2 - i].cols * sizeof(float),
        split_channels[2 - i].rows,
        cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    noncontiguousWrapper.release();
  }

  void SAMTile::get_tile_data(CompositeVoronoi *_comp, unsigned int _interval) {
    for (int xx = 0; xx < _interval; ++xx) {
      for (int yy = 0; yy < _interval; ++yy) {
        auto gMat = _comp->imagePyramid->level[0]->getTile(location.x + xx, location.y + yy);
        set_component_tile({location.x + xx - 1, location.y + yy - 1}, {xx, yy}, gMat);
      }
    }
  }


  void SAMTile::on_click() {
    priority = 0;
    for (auto &p: neighbors) {
      p.first->priority = 1;
    }
  }


  void AccessSAM::initialize() {
    cudaSetDevice(parent->compositorCudaDevice);
    int interval = (parent->SAMTileSize / parent->tileSize);
    assert(interval % 2 == 0);
    auto comp = parent->composites[0];

    auto ul = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto lr = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));


    int id = 0;
    int numSAMTiles = ceil((lr.x - ul.x + 1) / (interval - 1)) * ceil((lr.y - ul.y + 1) / (interval - 1));
    //size_t nBytesPerImage = 4 * 3 * parent->SAMTileSize * parent->SAMTileSize;
    //cudaMalloc(&batchImageEmbedBuffer,numSAMTiles * nBytesPerImage);

    int yTileCount = 0;
    for (int y = ul.y; y <= lr.y; ++y) {
      if ((yTileCount - 1) % (interval - 1) == 0) {
        int xTileCount = 0;

        for (int x = ul.x; x <= lr.x; ++x) {
          if ((xTileCount - 1) % (interval - 1) == 0) {
            auto st = new SAMTile(id, {x, y}, parent->SAMTileSize);
            tiles.push_back(st);
            ++id;

            //add links to neighbors
            if (xTileCount > 0) {
              auto brotherX = tiles[get_tile_id({x - interval + 1, y}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int xx = 0; xx < interval; ++xx) {
                temp.emplace_back(x + xx, y);
              }
              tiles.back()->neighbors.emplace_back(brotherX, temp);
              brotherX->neighbors.emplace_back(tiles.back(), temp);
            }
            if (yTileCount > 0) {
              auto brotherY = tiles[get_tile_id({x, y - interval + 1}, comp->componentIndex)];
              std::vector<Point2i> temp;
              for (int yy = 0; yy < interval; ++yy) {
                temp.emplace_back(x, y + yy);
              }
              tiles.back()->neighbors.emplace_back(brotherY, temp);
              brotherY->neighbors.emplace_back(tiles.back(), temp);
            }
            if (yTileCount > 0 && xTileCount > 0) {
              auto brotherXY = tiles[get_tile_id({x - interval + 1, y - interval + 1}, comp->componentIndex)];
              tiles.back()->neighbors.push_back({brotherXY, {{x, y}}});
              brotherXY->neighbors.push_back({tiles.back(), {{x, y}}});
            }
          }
          ++xTileCount;
        }
      }
      ++yTileCount;
    }
    //
    // int numImg = 5;
    // void* imageData = malloc(numImg * nBytesPerImage);
    // cudaMemcpy(imageData,batchImageEmbedBuffer + 50 * nBytesPerImage,numImg * nBytesPerImage,cudaMemcpyDeviceToHost);
    // std::ofstream out("/media/max/Data/pathcam_SAM/imagearray", std::ios::binary);
    // if (!out) {
    //   throw std::runtime_error("Failed to open file for writing: ");
    // }
    // out.write(reinterpret_cast<const char*>(imageData), numImg*nBytesPerImage);
    // out.close();
    // int k = 0;
  }

  void AccessSAM::load_model() {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "sam2_embedder");
    Ort::SessionOptions session_options;

    auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, parent->compositorCudaDevice);
    if (status) {
      const char *error_message = Ort::GetApi().GetErrorMessage(status);
      std::cerr << "Failed to add CUDA execution provider: " << error_message << std::endl;
      Ort::GetApi().ReleaseStatus(status);
    }

    session = new Ort::Session(env, parent->SAM_encoder_path.toString().c_str(), session_options);
  }


  void AccessSAM::embed_SAM_tiles() {
/*
    ProcessImage(parent->SAM_encoder_path.toString(),parent->SAM_decoder_path.toString(),
  "/media/max/Data/pathcam_SAM/test_trt/images",
  "/media/max/Data/pathcam_SAM/test_trt/labels",
  "/media/max/Data/pathcam_SAM/test_trt/",
  "fp32",64);
*/
    ProcessImage(parent->SAM_encoder_path.toString(),parent->SAM_decoder_path.toString(),
  "/home/max/Documents/sam2_trt_inference/sample_data/images",
  "/home/max/Documents/sam2_trt_inference/sample_data/bboxes",
  "/home/max/Documents/sam2_trt_inference/sample_data/",
  "fp32",64);

    tensorrt_common::BatchConfig batch_config_encoder = {1, 1, 1};
    tensorrt_common::BuildConfig build_config_encoder(
      "Entropy", -1, false, false, false, 0.0, false, {});
    const size_t max_workspace_size = 4ULL << 30;
    auto sie = SAM2ImageEncoder(parent->SAM_encoder_path.toString(), "fp32", batch_config_encoder, max_workspace_size,
                                build_config_encoder);



    std::deque<SAMTile *> embedQueue;
    for (auto &st: tiles) {
      embedQueue.push_back(st);
    }

    int decoder_batch_limit = 50;
    Size encoder_input_size(parent->SAMTileSize, parent->SAMTileSize);
    std::vector encoder_output_sizes = {
      sie.embed_size_, sie.feats_0_size_, sie.feats_1_size_
    };
    tensorrt_common::BatchConfig batch_config_decoder = {1, decoder_batch_limit / 2, decoder_batch_limit};
    tensorrt_common::BuildConfig build_config_decoder("Entropy",
                                                      -1,
                                                      false,
                                                      false,
                                                      false,
                                                      0.0,
                                                      false,
                                                      {});
    auto sid = SAM2ImageDecoder(parent->SAM_decoder_path.toString(),
                                "fp32",
                                batch_config_decoder,
                                max_workspace_size,
                                build_config_decoder,
                                encoder_input_size,
                                encoder_output_sizes);

    size_t nElementsPerChannel = parent->SAMTileSize * parent->SAMTileSize;
    auto comp = parent->composites[0];

    SAMTile *tileToFree = nullptr;
    int i = 0;
    while (!embedQueue.empty()) {
      std::sort(embedQueue.begin(), embedQueue.end(), tile_compare);
      auto tile = embedQueue.front();
      embedQueue.pop_front();

      tile->get_tile_data(comp, parent->SAMTileSize / parent->tileSize);
      cudaMalloc(&tile->rawBuffer, nElementsPerChannel * 3 * sizeof(float));
      tile->make_raw_buffer(tile->rawBuffer);

      cudaMalloc(&tile->feats_0_data_d_, nElementsPerChannel * 2 * sizeof(float));
      cudaMalloc(&tile->feats_1_data_d_, nElementsPerChannel * sizeof(float));
      cudaMalloc(&tile->embed_data_d_, nElementsPerChannel * sizeof(float));

      std::vector buffer{tile->rawBuffer, tile->embed_data_d_, tile->feats_1_data_d_, tile->feats_0_data_d_};

      CHECK_CUDA_ERROR(cudaStreamSynchronize(*sie.stream_));
      if (tileToFree) {
        cudaFree(tileToFree->rawBuffer);
      }
      sie.Infer(buffer);

      tileToFree = tile;

      if (tile->ID == 83) {
        std::vector<cv::Rect> box_coords = ReadAndTransformCoordinates("/media/max/Data/pathcam_SAM/SAM_bBox_ID83.txt");
      }
    }
    if (tileToFree) {
      cudaFree(tileToFree->rawBuffer);
    }
    int k = 0;
  }


  int AccessSAM::get_tile_id(Point2i _location, unsigned int _componentIndex) const {
    auto comp = parent->composites[_componentIndex];
    auto rootOffset = comp->imagePyramid->level[0]->getIJ(Point2f(comp->root_offset.x, comp->root_offset.y));
    auto maxOffset = comp->imagePyramid->level[0]->getIJ(Point2f(comp->max_offset.x, comp->max_offset.y));
    auto pt1 = (_location.y - rootOffset.y) / 3;
    auto pt2 = (maxOffset.x - rootOffset.x) / 3;
    auto pt3 = (_location.x - rootOffset.x) / 3;
    return pt1 * pt2 + pt3;
  }
}
