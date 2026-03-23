//
// Created by cooper on 5/9/25.
//
#include "pathCam.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
namespace pathCam {
#ifdef HAVE_OPENCV_CUDAARITHM


  void CompositeVoronoi::clean_face(std::vector<Point2i> &_face) {
    _face.push_back(_face[0]);
    int i = 1;
    while (i < _face.size()) {
      if (_face[i].x == _face[i - 1].x && _face[i].y == _face[i - 1].y) {
        _face.erase(_face.begin() + i);
      } else {
        i++;
      }
    }
    //ensure_clockwise(_face);
  }


  void CompositeVoronoi::ensure_clockwise(std::vector<Point2i> &_face) {
    double area = 0.0;
    for (int i = 1; i < _face.size(); ++i) {
      const cv::Point2i &p0 = _face[i - 1];
      const cv::Point2i &p1 = _face[i];
      area += (p0.x * p1.y - p1.x * p0.y);
    }
    if (area < 0) {
      std::reverse(_face.begin(), _face.end());
    } else {
      int k = 0;
    }
  }


  void CompositeVoronoi::make_meshgrid() {
    Mat x_row(1, imageSize.width, CV_16S);
    Mat y_col(imageSize.height, 1, CV_16S);

    for (int i = 0; i < imageSize.width; i++) {
      x_row.at<short>(i) = i;
    }
    for (int i = 0; i < imageSize.height; i++) {
      y_col.at<short>(i) = i;
    }

    Mat X, Y;
    repeat(x_row, imageSize.height, 1, X);
    repeat(y_col, 1, imageSize.width, Y);

    meshGridX.upload(X);
    meshGridY.upload(Y);

    diffGPU = cuda::GpuMat(imageSize, CV_32S);
    xp1 = cuda::GpuMat(imageSize, CV_32F);
    xp2 = cuda::GpuMat(imageSize, CV_32F);
    binaryCompare = cuda::GpuMat(imageSize, CV_8U);

    polyMaskGPU = cuda::GpuMat(imageSize, CV_8U);
  }


  void CompositeVoronoi::coopers_GPU_vectorized_convex_mask_maker(std::vector<Point2i> &_face) {
    polyMaskGPU.setTo(Scalar(255));
    for (int i = 1; i < _face.size(); i++) {
      int x0 = _face[i - 1].x;
      int x1 = _face[i].x;

      int y0 = _face[i - 1].y;
      int y1 = _face[i].y;


      //largest values to appear as x1 or y1 are about 400,000
      //meshGridX -> CV_16S
      //meshGridY -> CV_16S
      //diffGPU -> CV_32S
      //xp1 and xp2 -> CV_32F
      //polyMaskGPU and binaryCompare -> CV_8U

      //get the x component of the vector formed between our line and vector formed between
      //the base of our line and every point in the mat
      cuda::subtract(meshGridX, x0, diffGPU);

      //multiply x component of every vector by y component of line
      cuda::multiply(diffGPU, y1 - y0, xp2);

      //get the y component of the vector formed between our line and vector formed between
      //the base of our line and every point in the mat
      cuda::subtract(meshGridY, y0, diffGPU);

      //multiply y component of every vector by x component of line
      cuda::multiply(diffGPU, x1 - x0, xp1);

      //subtraction as defined by cross product forumula
      cuda::subtract(xp2, xp1, xp1);

      //set all values less than 0 to 0, all values greater than 0 to 255
      cuda::compare(xp1, 0, binaryCompare, CMP_GE);

      Mat temp;
      binaryCompare.download(temp);
      line(temp, _face[i - 1], _face[i], Scalar(150), 100);
      imwrite("/media/max/Data/binaryCompare.png", temp);

      cuda::multiply(polyMaskGPU, binaryCompare, polyMaskGPU);
    }
  }


  bool Composite::prepare_4CPA(Image *img, const std::vector<Point2i> &affectedTiles, const bool forceFullImage) {
    if (affectedTiles.size() < 100 && !forceFullImage) {
      bool ans = false;

      img->regInfo->accessMutex.lock();
      auto AbC = img->regInfo->absoluteCoords;
      img->regInfo->accessMutex.unlock();

      Rect imageBoxCompSpace(AbC, imageSize);
      img->buffer_mutex.lock();
      for (auto &tile: affectedTiles) {
        Rect tileBoxCompSpace(parent->tileSize * tile, Size(parent->tileSize, parent->tileSize));

        auto intersectionInCompSpace = tileBoxCompSpace & imageBoxCompSpace;

        if (intersectionInCompSpace.empty()) { continue; }

        auto intersectionInImageSpace = intersectionInCompSpace - AbC;
        
        ans = ans || prepare_4CPA(img, intersectionInImageSpace);
      }
      img->buffer_mutex.unlock();
      return ans;
    }

    img->buffer_mutex.lock();
    bool ans = prepare_4CPA(img);
    img->buffer_mutex.unlock();
    return ans;
  }



  bool Composite::prepare_4CPA(Image *img, Rect roi_) {
    assert(roi_.x >= 0 && roi_.y >= 0);
    bool wholeImage = false;

    try {
      if (roi_.width * roi_.height == 0) {
        roi_ = Rect(0, 0, imageSize.width, imageSize.height);
        wholeImage = true;
      }

      if (!parent->unifiedMemory) {
        //wait for buffer to be on gpu
        std::unique_lock lock(img->cudaBufferMutex);
        img->cudaBufferConVar.wait(lock, [&] { return img->cudaBufferReady; });

        //build and debayer with gpumat objects
        cuda::GpuMat rawMat(imageSize, CV_8U, img->get_raw_cuda());
        cuda::cvtColor(rawMat, threeChannelPrealGPU, COLOR_BayerBG2BGR, 0, parent->cvCompositeStream);
        wholeImage = true;
      } else {
        assert(img->get_Raw());

        adjust_roi_for_debayer(roi_);
        Mat rawMat(imageSize, CV_8U, img->get_Raw());
        cvtColor(rawMat(roi_), threeChannelPreallocated(roi_), COLOR_BayerBG2BGR);
      }

      //ff correct
      if (convertHoldingGPU.empty()) {
        convertHoldingGPU = cuda::GpuMat(imageSize,CV_32FC3);
      }
      threeChannelPrealGPU(roi_).convertTo(convertHoldingGPU(roi_), CV_32F, parent->cvCompositeStream);
      if (flatfieldKnown) {
        cuda::divide(convertHoldingGPU(roi_), ffGPU(roi_),
                     convertHoldingGPU(roi_), 1, CV_32F, parent->cvCompositeStream);
      }

      convertHoldingGPU(roi_).convertTo(threeChannelPrealGPU(roi_), CV_8UC3, parent->cvCompositeStream);

      //add alpha channel
      cuda::split(threeChannelPrealGPU(roi_), channelsGPU, parent->cvCompositeStream);
      channelsGPU.push_back(rectMaskGPU(roi_));
      cuda::merge(channelsGPU, fourChannelPrealGPU(roi_), parent->cvCompositeStream);

      parent->cvCompositeStream.waitForCompletion();
    }catch (...) {
      int k = 0;
    }
    return wholeImage;
  }

void CompositeVoronoi::update() {
    if (suspended) { return; }

    //place component in MR image
    if (imagePyramid->scale == 0 && !xcMatchInitiated) {
      if (staging.empty()) { return; }
      xcInProgress = true;
      xcMatchInitiated = true;

      std::thread t([this, img = staging.front()->image]() {
        std::lock_guard lock(EstRoot_mutex);
        std::cout << "component " << componentIndex << " establishing scale on separate thread" << std::endl;
        establish_scale_at_root(img);
        xcInProgress = false;
      });
      t.detach();
    }

    // PROCESS NEW FRAMES BEGIN
    if (!staging.empty()) {
      auto ri = staging.front();
      auto img = ri->image;
      memberFrames.push_back(img);
      assert(img->regInfo);

      staging.pop();
      ++frameCount;

      if (img->labelObserved) {
        ++observedLabels[img->label];
        int maxObservations = 0;
        unsigned winner = componentMagLabel;
        for (auto &[label,observationCount]: observedLabels) {
          if (observationCount > maxObservations) {
            maxObservations = observationCount;
            winner = label;
          }
        }
        if (winner != componentMagLabel) {
          componentMagLabel = winner;

          Poco::FastMutex::ScopedLock lock(update_mutex);
          get_flatfield();
          imagePyramid->set_mag_label(componentMagLabel);
          parent->MRImageSet->sort_by_scale();
        }
      }

      update_Bbox_no_composite({ri});
      expand_subdiv({ri});

      Point2i distToLA = ri->absoluteCoords - lastAccepted;
      if (distToLA.x * distToLA.x + distToLA.y * distToLA.y >= minDistance) {
        //add point to delaunay triangulation
        std::vector<Point2i> face;
        auto fShift = Point2f(ri->absoluteCoords.x, ri->absoluteCoords.y);
        auto res = add_point_to_delaunay_triangulation(fShift, img, face, false);

        if (res >= 0) {
          //image accepted
          if (!parent->unifiedMemory) {
            std::unique_lock<std::mutex> lock(img->cudaBufferMutex);
            if (!img->raw_buffer_cuda) {
              assert(img->get_Raw());
              const size_t nBytes = static_cast<size_t>(img->height) * img->width;

              // cudaMemcpyAsync requires a pinned source for a truly async transfer.
              // Copy raw_buffer into a pinned staging buffer before launching.
              char *pinnedBuf = nullptr;
              cudaMallocHost(reinterpret_cast<void **>(&pinnedBuf), nBytes);
              memcpy(pinnedBuf, img->get_Raw(), nBytes);

              cudaMallocAsync(reinterpret_cast<void **>(&img->raw_buffer_cuda), nBytes,
                              cuda::StreamAccessor::getStream(parent->cvCompositeStream));
              cudaMemcpyAsync(img->raw_buffer_cuda, pinnedBuf, nBytes, cudaMemcpyHostToDevice,
                              cuda::StreamAccessor::getStream(parent->cvCompositeStream));

              // After the stream work completes: free the pinned staging buffer and
              // signal cudaBufferReady so consumers waiting on cudaBufferConVar unblock.
              struct Ctx {
                Image *img;
                char *pinnedBuf;
              };
              auto *ctx = new Ctx{img, pinnedBuf};
              cudaLaunchHostFunc(cuda::StreamAccessor::getStream(parent->cvCompositeStream),
                                 [](void *ud) {
                                   auto *c = static_cast<Ctx *>(ud);
                                   cudaFreeHost(c->pinnedBuf);
                                   {
                                     std::lock_guard<std::mutex> cbLock(c->img->cudaBufferMutex);
                                     c->img->cudaBufferReady = true;
                                     c->img->cudaBufferConVar.notify_one();
                                   }
                                   delete c;
                                 }, ctx);
            }
          }
          img->vertexId = res;
          contributingFrames.insert(img);
          lastAccepted = ri->absoluteCoords;

          if (!img->subsequentMatchLaunched) {
            img->load_raw_from_disk(); //freed in ComponentMatchSearch::run()
            img->subsequentMatchLaunched = true;
            ++outstandingCMS_jobs;
            const auto cms = new ComponentMatchSearch(parent, img, this);
            parent->jqSecondary->add_runnable(cms);
          }


          std::vector<Point2i> effectedTiles;
          std::vector<Point2i> effectedTilesNoMask;
          bool noMask = false;

          //calculate region of pyramid for data placement
          auto imageBox = cv::Rect_<float>(ri->absoluteCoords.x, ri->absoluteCoords.y, img->width,
                                           img->height);

          if (componentMagLabel == Image::_2X) {
            calculate_effected_tiles_round(face, effectedTiles, ri->absoluteCoords);
          } else {
            noMask = true;
            calculate_effected_tiles(face, effectedTiles, ri->absoluteCoords, &effectedTilesNoMask);

          }

          auto tiles = effectedTilesNoMask;
          tiles.insert(tiles.end(),effectedTiles.begin(),effectedTiles.end());
          prepare_4CPA(img, tiles);
          if (!effectedTilesNoMask.empty()) {
            imagePyramid->insertTilesAtBase(fourChannelPreallocated,rectMask,imageBox,effectedTilesNoMask);
          }
          imagePyramid->insertTilesAtBase(fourChannelPreallocated, polyMaskOutput, imageBox, effectedTiles);
        }
      }
      img->free_memory_RAW();

      float x = (imagePyramid->offset.x + img->regInfo->absoluteCoords.x) * imagePyramid->scale;
      float y = (imagePyramid->offset.y + img->regInfo->absoluteCoords.y) * imagePyramid->scale;
      float w = parent->image_width * imagePyramid->scale;
      float h = parent->image_height * imagePyramid->scale;
      bool showAsCircle = (componentMagLabel == Image::_2X);

      parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                Image::get_label(componentMagLabel), get_scale());
    }
  }
  void CompositeVoronoi::self_reset() {
    imagePyramid->level[0]->resetEdges(Point2i(root_offset.x, root_offset.y), Point2i(max_offset.x, max_offset.y));
    subdiv_Bbox = Bbox(-50000, -50000, 50000, 50000);
    subdiv.initDelaunay(subdiv_Bbox.as_cvRect());
    composite.release();
    memberImages.clear();
    matchedEdges.clear();
    delaunayMembers.clear();
    root_offset = Point2i(0, 0);
    max_offset = Point2i(0, 0);
  }


  void CompositeVoronoi::expand_subdiv(std::vector<RegInfo *> new_info) {
    bool extend = false;
    if (root_offset.x < subdiv_Bbox.min_x) {
      extend = true;
      subdiv_Bbox.min_x *= 2;
    }
    if (root_offset.y < subdiv_Bbox.min_y) {
      extend = true;
      subdiv_Bbox.min_y *= 2;
    }
    if (max_offset.x > subdiv_Bbox.max_x) {
      extend = true;
      subdiv_Bbox.max_x *= 2;
    }
    if (max_offset.y > subdiv_Bbox.max_y) {
      extend = true;
      subdiv_Bbox.max_y *= 2;
    }
    if (extend) {
      std::vector<std::vector<Point2f> > facets;
      std::vector<Point2f> centers;

      subdiv.getVoronoiFacetList({}, facets, centers);
      subdiv = Subdiv2D(subdiv_Bbox.as_cvRect());
      subdiv.insert(centers);
    }
  }


  void CompositeVoronoi::notify_job_complete() {
    jobCount--;
    if (jobCount == 0) {
      wakeEvent.set();
    }
  }

  void CompositeVoronoi::rebuild_DT_elementwise(std::vector<RegInfo *> new_info, bool forceAdd, bool shuffle) {
    if (new_info.size() > 1 && shuffle) {
      // this shuffle is very important for reducing image count in the DT.
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(new_info), std::end(new_info), rng);
    }

    for (auto ni: new_info) {
      std::vector<Point2i> face;
      auto fShift = Point2f(ni->absoluteCoords.x, ni->absoluteCoords.y);
      auto img = parent->get_image_ref((*ni).index);
      img->absoluteCoords = Point2i(ni->absoluteCoords.x, ni->absoluteCoords.y);
      auto res = add_point_to_delaunay_triangulation(fShift, img, face, forceAdd);

      update_Bbox_no_composite({ni});
      expand_subdiv({ni});
      freshMask.copyTo(polyMaskOutput);
    }
  }

  int CompositeVoronoi::add_point_to_delaunay_triangulation_with_adjustment(
    cv::Point2f _point, pathCam::Image *_image,
    std::vector<Point2i> &_face,
    bool _forceAdd) {
    //make copy of subdiv incase we decide not to use new point
    Subdiv2D tempSubdiv(subdiv);

    //correct registration in case matchedTo was corrected at composite time
    //_image->correct_registration({});
    auto point = cv::Point2f(_image->regInfo->absoluteCoords.x, _image->regInfo->absoluteCoords.y);
    //add new point
    int vertxId = subdiv.insert(point);

    //gather adjacent images for correcting _image registration
    std::vector<unsigned long> adjacentVerts;
    if (delaunayMembers.size() > 2) {
      //int firstEdge = subdiv.vtx[vertxId].firstEdge;
      int firstEdge = -1;
      subdiv.getVertex(vertxId, &firstEdge);
      int nextEdge = firstEdge;
      do {
        nextEdge = subdiv.nextEdge(nextEdge);
        auto val = delaunayMembers.find(subdiv.edgeDst(nextEdge));
        if (val != delaunayMembers.end()) {
          adjacentVerts.push_back(val->second);
        }
      } while (nextEdge != firstEdge);
      //_image->correct_registration(adjacentVerts);
      subdiv = tempSubdiv;
      point = cv::Point2f(_image->regInfo->absoluteCoords.x, _image->regInfo->absoluteCoords.y);
      vertxId = subdiv.insert(point);
    }

    //get voronoi facets for only this face
    std::vector<std::vector<Point2f> > facets;
    std::vector<Point2f> centers;
    subdiv.getVoronoiFacetList({vertxId}, facets, centers);


    //shift and recast
    for (auto &ii: facets[0]) {
      //we have pulled only one face so facets has only 1 element
      ii.x -= centers[0].x;
      ii.x += imageSize.width / 2;
      ii.y -= centers[0].y;
      ii.y += imageSize.height / 2;
      _face.push_back((Point2i) ii);
    }

    //build polygon mask for new point
    cv::fillConvexPoly(polyMaskOutput, _face, cv::Scalar(255));

    if (!_forceAdd) {
      //test for exclusion of frame via rollback
      int nonzeroMin;
      if (_image->label == Image::_2X) {
        polyMaskOutput = polyMaskOutput.mul(circleMask);
        nonzeroMin = parent->scope_radius * parent->scope_radius * 3.14 * 0.10;
      } else {
        nonzeroMin = _image->width * _image->height * 0.1;
      }

      if (countNonZero(polyMaskOutput) <= nonzeroMin) {
        //contributing less than x% of its pixels, revert and don't bother loading from disk
        subdiv = tempSubdiv;
        _image->free_memory_RAW();
        memberImages.push_back({_image, false});
        return -1;
      }
    }
    _image->absoluteCoords = point;
    memberImages.push_back({_image, true});
    delaunayMembers.insert({vertxId, _image->index});
    return vertxId;
  }

  int CompositeVoronoi::add_point_to_delaunay_triangulation(cv::Point2f _point, pathCam::Image *_image,
                                                            std::vector<Point2i> &_face, bool _forceAdd,
                                                            bool _drawMask) {
    if (!_forceAdd) {
      _drawMask = true;
    }

    //make copy of subdiv incase we decide not to use new point
    Subdiv2D tempSubdiv(subdiv);

    //add new point
    int vertxId = subdiv.insert(_point);
    //auto image = parent->get_image_ref(_image_index);

    //get voronoi facets for only this face
    std::vector<std::vector<Point2f> > facets;
    std::vector<Point2f> centers;
    subdiv.getVoronoiFacetList({vertxId}, facets, centers);


    //shift and recast
    for (auto &ii: facets[0]) {
      //we have pulled only one face so facets has only 1 element
      ii.x -= centers[0].x;
      ii.x += imageSize.width / 2;
      ii.y -= centers[0].y;
      ii.y += imageSize.height / 2;
      _face.push_back((Point2i) ii);
    }

    // clean_face(_face);

    if (_drawMask) {
      polyMaskOutput.setTo(Scalar(0));
      fillConvexPoly(polyMaskOutput, _face, cv::Scalar(255));
    }

    //test for exclusion of frame via rollback
    int nonzeroMin;
    if (componentMagLabel == Image::_2X) {
      polyMaskOutput = polyMaskOutput.mul(circleMask);
      nonzeroMin = parent->scope_radius * parent->scope_radius * 3.14 * 0.2;
    } else {
      nonzeroMin = _image->width * _image->height * 0.25;
    }

    if (!_forceAdd && countNonZero(polyMaskOutput) <= nonzeroMin) {
      //contributing less than x% of its pixels, revert and don't bother loading from disk
      subdiv = tempSubdiv;
      return -1;
    }

    auto ret = delaunayMembers.insert({vertxId, _image->index});
    if (!ret.second) {
      subdiv = tempSubdiv;
      return -1;
    }


    return vertxId;
  }


  void CompositeVoronoi::add_images_no_composite(std::vector<RegInfo *> new_info, bool _force_add) {
    auto start = std::chrono::high_resolution_clock::now();

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<long> indexes;
    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i]->index);
    }
    std::vector<Image *> images = parent->get_image_ref(indexes);
    bool update = false;

    for (int i = 0; i < images.size(); i++) {
      // if(indexes[i] == lastAcceptedImageIndex){
      //   continue;
      // }
      // lastAcceptedImageIndex = indexes[i];

      //correct placement of last added image
      //      if (!memberImages.empty() && !_force_add) {
      //        //put in reverse match runnable
      //        auto rmr = new ReverseMatchRunnable(parent, images[i]->index, lastAcceptedImageIndex);
      //        jobCount++;
      //        parent->JobQ->add_runnable(rmr);
      //        wakeEvent.wait();
      //      }


      //add point to delaunay triangulation
      std::vector<Point2i> face;
      auto fShift = Point2f(new_info[i]->absoluteCoords.x, new_info[i]->absoluteCoords.y);
      auto res = add_point_to_delaunay_triangulation(fShift, images[i], face, _force_add);

      //res is {vertexId,maskId}
      if (res == -1) {
        continue;
      }
      update = true;
      images[i]->vertexId = res;

      //indicate that a new image has been added since last global alignment
      needsAlignment = true;

      //build image with alpha channel
      images[i]->load_raw_from_disk();
      Mat image_Mat = cv::Mat(imageSize, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat, threeChannelPreallocated, COLOR_BayerBG2BGR);
      images[i]->free_memory_RAW();
      if (componentMagLabel != 0) {
        //flatfield correct
        divide(threeChannelPreallocated, ff, convertHolding, 1, CV_32F);
        //brighten
        cv::pow(convertHolding, 1.1, convertHolding);
        convertHolding.convertTo(threeChannelPreallocated, CV_8UC3);
      }
      channels[0] = threeChannelPreallocated; //3 channel


      //add alpha channel
      channels[1] = rectMask;
      merge(channels, fourChannelPreallocated);

      //calculate effected tiles
      std::vector<Point2i> effectedTiles;
      std::vector<Point2i> effectedTiles2;
      std::vector<Point2i> effectedTilesNoMask;


      auto imageBox = cv::Rect_<float>(images[i]->absoluteCoords.x, images[i]->absoluteCoords.y, images[i]->width,
                                       images[i]->height);

      if (componentMagLabel == Image::_2X) {
        calculate_effected_tiles_round(face, effectedTiles, images[i]->absoluteCoords);
      } else {
        calculate_effected_tiles(face, effectedTiles, images[i]->absoluteCoords, &effectedTilesNoMask);
#ifndef HAVE_OPENCV_CUDAARITHM
        imagePyramid->insertTilesAtBase(fourChannelPreallocated, Mat(), imageBox, effectedTilesNoMask);
#endif
      }
#ifndef HAVE_OPENCV_CUDAARITHM
      imagePyramid->insertTilesAtBase(fourChannelPreallocated, polyMaskOutput, imageBox, effectedTiles);
#endif
      if (parent->inferencing) {
        std::vector<Point2i> tiles;
        tiles.reserve(effectedTiles.size() + effectedTilesNoMask.size());
        tiles.insert(tiles.end(), effectedTiles.begin(), effectedTiles.end());
        tiles.insert(tiles.end(), effectedTilesNoMask.begin(), effectedTilesNoMask.end());

        auto pushForInferencing = push_for_inferencing(tiles);
        parent->push_tile_embed_Q(pushForInferencing, componentIndex);
      }
      //update pyramid bounds, reset mask
      imagePyramid->bounds = imagePyramid->level[0]->bounds;
      polyMaskOutput = freshMask.clone();
    }

    //highlight bounds of last frame
    if (imagePyramid->scale > 0 && update) {
      float x = (imagePyramid->offset.x + images.back()->absoluteCoords.x) * imagePyramid->scale;
      float y = (imagePyramid->offset.y + images.back()->absoluteCoords.y) * imagePyramid->scale;
      float w = parent->image_width * imagePyramid->scale;
      float h = parent->image_height * imagePyramid->scale;
      bool showAsCircle = (componentMagLabel == Image::_2X);

      parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                Image::get_label(componentMagLabel), get_scale());

      parent->notify_observers();

      //      auto stop = std::chrono::high_resolution_clock::now();
      //      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
      //      if (duration < 250) {
      //        Poco::Thread::sleep(250 - duration);
      //      }
    }
  }

  std::vector<Point2i> CompositeVoronoi::push_for_inferencing(std::vector<Point2i> &_tiles) {
    std::vector<Point2i> temp;
    std::remove_copy_if(queuedTiles.begin(), queuedTiles.end(), std::back_inserter(temp),
                        [&_tiles](const Point2i &arg) {
                          return (std::find(_tiles.begin(), _tiles.end(), arg) != _tiles.end());
                        });

    queuedTiles = _tiles;
    return temp;
  }



  void CompositeVoronoi::calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result,
                                                  Point2f absCoord, std::vector<Point2i> *additionalResult) {
    std::vector<Point2i> tileIndices;
    std::map<int, std::vector<float> > tilesByColumn;

    //get the tile column of the left and right edges
    int columnBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).x;
    int columnBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x + imageSize.width, absCoord.y)).x;

    //get the tile row top and bottom edges of the image frame
    long rowBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).y;
    long rowBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x, absCoord.y + imageSize.height)).y;

    //get the tile row top and bottom as if the image was square. This is used to prevent chevrons
    auto dimensionDifference = parent->image_width - parent->image_height;
    long rowBoundFalseLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y - dimensionDifference / 2))
        .y;
    long rowBoundFalseHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x, absCoord.y + imageSize.height + dimensionDifference / 2)).y;

    float yPixelBoundLow = float(rowBoundFalseLow) * float(imagePyramid->tile_size);
    float yPixelBoundHigh = float(rowBoundFalseHigh) * float(imagePyramid->tile_size);

    //for each edge of the voronoi mask
    for (int ii = 0; ii < maskAsPolygon.size() ; ii++) {
      //if we are at the last point, make the next point the first point (this makes the last edge)
      int ii2 = (ii + 1) == maskAsPolygon.size() ? 0 : ii + 1;
      // int ii2 = ii + 1;

      //create an Point2f from the cv::Point2i and shift voronoi mask to tile space
      auto p1 = Point2f(maskAsPolygon[ii].x + absCoord.x, maskAsPolygon[ii].y + absCoord.y);
      auto p2 = Point2f(maskAsPolygon[ii2].x + absCoord.x, maskAsPolygon[ii2].y + absCoord.y);

      //test for duplicate points that result from the voronoi calculation
      if (p1.x == p2.x && p1.y == p2.y) {
        continue;
      }

      //retrieve the tiles these points fall within
      auto tile1 = imagePyramid->level[0]->getIJ(p1);
      auto tile2 = imagePyramid->level[0]->getIJ(p2);

      //get the column of these tiles
      int column1 = tile1.x;
      int column2 = tile2.x;

      //determine which column is on the right and which is on the left
      int xlow = min(column1, column2);
      int xhigh = max(column1, column2);

      //take the column range bounds to be the most inward of the frame edges and the voronoi cell vertices
      //this accomplishes the same thing as taking the polygon intersection of the voronoi face and the image frame
      if (columnBoundHigh < xlow) {
        continue; //no intersection with frame tiles
      }
      if (columnBoundLow > xhigh) {
        continue;
      }
      xlow = max(columnBoundLow, xlow);
      xhigh = min(columnBoundHigh, xhigh);

      auto plow = p1.x < p2.x ? p1 : p2;
      auto phigh = p1.x >= p2.x ? p1 : p2;

      for (float j = xlow; j <= xhigh; j++) {
        float columnLeftEdge = j * float(imagePyramid->level[0]->getTileSize());
        float columnRightEdge = (j + 1) * float(imagePyramid->level[0]->getTileSize());
        float xloclow = max(columnLeftEdge, plow.x);
        float xlochigh = min(columnRightEdge, phigh.x);

        long enterColumn = segment_yval_at_point(xloclow, p1, p2);
        long exitColumn = segment_yval_at_point(xlochigh, p1, p2);

        tilesByColumn[j].push_back((float) enterColumn);
        tilesByColumn[j].push_back((float) exitColumn);
      }
    }

    auto maxDist = std::sqrt(std::pow(parent->image_height / 2, 2) + std::pow(parent->image_width / 2, 2));
    auto columnBoundFalseLow = imagePyramid->level[0]->getIJ(
      Point2f(double(parent->image_width / 2) + absCoord.x - maxDist, absCoord.y)).x;
    auto columnBoundFalseHigh = imagePyramid->level[0]->getIJ(
      Point2f(double(parent->image_width / 2) + absCoord.x + maxDist, absCoord.y)).x;
    for (int x = columnBoundLow; x <= columnBoundHigh; x++) {
      bool columnIntersectsMask = tilesByColumn.find(x) != tilesByColumn.end();
      int lastTile, firstTile;

      if (columnIntersectsMask) {
        float firstPoint_y = *std::min_element(tilesByColumn[x].begin(), tilesByColumn[x].end());
        float lastPoint_y = *std::max_element(tilesByColumn[x].begin(), tilesByColumn[x].end());

        firstPoint_y = std::max(firstPoint_y, absCoord.y);
        lastPoint_y = std::min(lastPoint_y, absCoord.y + parent->image_height);

        lastTile = imagePyramid->level[0]->getIJ(
          Point2f(x * imagePyramid->level[0]->getTileSize(), lastPoint_y)).y;
        firstTile = imagePyramid->level[0]->getIJ(
          Point2f(x * imagePyramid->level[0]->getTileSize(), firstPoint_y)).y;
      }

      for (int y = rowBoundFalseLow; y <= rowBoundFalseHigh; y++) {
        auto tilePoint = Point2i(x, y);

        auto loc = std::find(falselyClaimedTiles.begin(), falselyClaimedTiles.end(), tilePoint);

        if (loc != falselyClaimedTiles.end()) {
          additionalResult->push_back(tilePoint);
          if (x > columnBoundLow && x < columnBoundHigh && y > rowBoundLow && y < rowBoundHigh) {
            falselyClaimedTiles.erase(loc);
          }
        } else if (columnIntersectsMask) {
          bool canAdd = false;
          for (int idx = 0; idx < 4; idx++) {
            int x = tilePoint.x * imagePyramid->tile_size + (idx % 2 == 0 ? imagePyramid->tile_size : 0);
            int y = tilePoint.y * imagePyramid->tile_size + (idx % 3 == 0 ? imagePyramid->tile_size : 0);

            if (abs(absCoord.x + imageSize.width / 2 - x) < imageSize.width / 2
                && abs(absCoord.y + imageSize.height / 2 - y) < imageSize.height / 2) {
              canAdd = true;
              break;
            }
          }

          if (canAdd) {
            if (y > firstTile && y < lastTile) {
              result.push_back(tilePoint);
            } else if (y == firstTile || y == lastTile) {
              result.push_back(tilePoint);
              falselyClaimedTiles.push_back(tilePoint);
              if (tilePoint.x < minTilex) { minTilex = tilePoint.x; }
              if (tilePoint.x > maxTilex) { maxTilex = tilePoint.x; }
              if (tilePoint.y < minTiley) { minTiley = tilePoint.y; }
              if (tilePoint.y > maxTiley) { maxTiley = tilePoint.y; }
            } else {
              falselyClaimedTiles.push_back(tilePoint);
            }
          }
        }
      }
    }
  }
  void CompositeVoronoi::GPU_add_images_no_composite(std::vector<RegInfo *> _newInfo, bool _force_add) {
    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<long> indexes;
    for (int i = 0; i < _newInfo.size(); i++) {
      indexes.push_back(_newInfo[i]->index);
    }


    std::vector<Image *> images = parent->get_image_ref(indexes);
    bool update = false;
    bool rootFound = false;

    for (int i = 0; i < images.size(); i++) {
      //images[i]->component_membership = componentIndex;
      assert(images[i]->regInfo->component_membership == componentIndex);

      if (images[i]->regInfo->root && !images[i]->regInfo->rootOfRoot) {
        rootFound = true;
        assert(parent->lastViewedFrame->get_raw_cuda() || parent->lastViewedFrame->get_Raw());
      }
      //add point to delaunay triangulation
      std::vector<Point2i> face;
      auto fShift = Point2f(_newInfo[i]->absoluteCoords.x, _newInfo[i]->absoluteCoords.y);
      auto res = add_point_to_delaunay_triangulation(fShift, images[i], face, _force_add);

      //res is {vertexId,maskId}
      if (res == -1) {
        continue;
      }

      contributingRegInfos.push_back(_newInfo[i]);
      contributingImages.insert(images[i]);

      update = true;
      images[i]->vertexId = res;

      if (parent->unifiedMemory) {
        polyMaskGPU = cuda::GpuMat(polyMaskOutput.rows, polyMaskOutput.cols,CV_8U, polyMaskOutput.data);
      } else {
        polyMaskGPU.upload(polyMaskOutput, parent->cvCompositeStream);
      }

      //indicate that a new image has been added since last global alignment
      needsAlignment = true;


      if (!parent->unifiedMemory) {
        //wait for buffer to be on gpu
        std::unique_lock lock(images[i]->cudaBufferMutex);
        images[i]->cudaBufferConVar.wait(lock, [&] { return images[i]->cudaBufferReady; });

        //build and debayer with gpumat objects
        cuda::GpuMat rawMat;
        rawMat = cuda::GpuMat(imageSize, CV_8U, images[i]->get_raw_cuda());
        cuda::cvtColor(rawMat, threeChannelPrealGPU, COLOR_BayerBG2BGR, 0, parent->cvCompositeStream);
      } else {
        Mat rawMat;
        rawMat = Mat(imageSize, CV_8U, images[i]->get_Raw());
        cvtColor(rawMat, threeChannelPreallocated, COLOR_BayerBG2BGR);
        threeChannelPrealGPU = cuda::GpuMat(imageSize,CV_8UC3, threeChannelPreallocated.data);
      }

      if (rootFound) {
        parent->cvCompositeStream.waitForCompletion();
        establish_scale_at_root(images[i]);
      }

      ff_correct_and_brighten();

      //get sift data and push it to sift ft extraction gpu
      //images[i]->siftData = GPU_extract_SIFT(threeChannelPrealGPU, 10000);
      // images[i]->siftInitialized = true;

      //auto newOverlaps = calculate_new_overlaps();
      //parent->push_SIFT_matches(newOverlaps, images[i]);
      // if (_newInfo[i]->root && !newOverlaps.empty()) {
      //   wakeEvent.wait();
      //   ff_correct_and_brighten();
      // }

      //add alpha channel
      cuda::split(threeChannelPrealGPU, channelsGPU, parent->cvCompositeStream);
      channelsGPU.push_back(rectMaskGPU);
      cuda::merge(channelsGPU, fourChannelPrealGPU, parent->cvCompositeStream);


      //calculate effected tiles
      std::vector<Point2i> effectedTiles;
      std::vector<Point2i> effectedTilesNoMask;

      //calculate region of pyramid for data placement
      auto imageBox = cv::Rect_<float>(images[i]->absoluteCoords.x, images[i]->absoluteCoords.y, images[i]->width,
                                       images[i]->height);

      if (componentMagLabel == Image::_2X) {
        calculate_effected_tiles_round(face, effectedTiles, images[i]->absoluteCoords);
      } else {
        calculate_effected_tiles(face, effectedTiles, images[i]->absoluteCoords, &effectedTilesNoMask);
        imagePyramid->insertTilesAtBase(fourChannelPrealGPU, rectMaskGPU, imageBox, effectedTilesNoMask);
      }
      parent->cvCompositeStream.waitForCompletion();

      auto t1 = std::chrono::high_resolution_clock::now();
      // imagePyramid->insertTilesAtBase(fourChannelPrealGPU, polyMaskGPU, imageBox, effectedTiles);

      imagePyramid->insertTilesAtBase(fourChannelPreallocated, polyMaskOutput, imageBox, effectedTiles);
      tileupwardsTime += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t1).
          count();

      if (parent->inferencing) {
        std::vector<Point2i> tiles;
        tiles.reserve(effectedTiles.size() + effectedTilesNoMask.size());
        tiles.insert(tiles.end(), effectedTiles.begin(), effectedTiles.end());
        tiles.insert(tiles.end(), effectedTilesNoMask.begin(), effectedTilesNoMask.end());

        auto pushForInferencing = push_for_inferencing(tiles);
        parent->push_tile_embed_Q(pushForInferencing, componentIndex);
      }

      //update pyramid bounds, reset mask

      polyMaskOutput.setTo(Scalar(0));
    }


    //highlight bounds of last frame
    if (imagePyramid->scale > 0 && update) {
      float x = (imagePyramid->offset.x + images.back()->absoluteCoords.x) * imagePyramid->scale;
      float y = (imagePyramid->offset.y + images.back()->absoluteCoords.y) * imagePyramid->scale;
      float w = parent->image_width * imagePyramid->scale;
      float h = parent->image_height * imagePyramid->scale;
      bool showAsCircle = (componentMagLabel == Image::_2X);

      parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                Image::get_label(componentMagLabel), get_scale());

      parent->notify_observers();
    }
  }

  void CompositeVoronoi::align_and_rebuild() {
    auto start = std::chrono::high_resolution_clock::now();


    for (auto &loser : absorbedComponents) {
      ftg->storedMatches.insert(loser->ftg->storedMatches.begin(), loser->ftg->storedMatches.end());
      for (auto &img : loser->landmarkFrames) {
        landmarkFrames.push_back(img);
      }
      loser->root->regInfo->root = false;
    }


    auto ig = ImageGraph();

    std::unordered_set<Image *> members = contributingFrames;
    members.insert(root);
    for (auto img : landmarkFrames) {
      members.insert(img);
    }
    auto matches = ftg->storedMatches; //matches are just stored here before being processed all at once.
    std::unordered_map<Image *, std::vector<std::shared_ptr<Match> > > adjacency;

    for (auto &m: matches) {
      adjacency[m->image_1].push_back(m);
      adjacency[m->image_2].push_back(m);
    }
    std::queue<Image*> q;

    // Seed with confirmed members
    for (auto img : members) {
      img->regInfo->component_membership = componentIndex;
      q.push(img);
    }

    while (!q.empty()) {
      Image *img = q.front();
      q.pop();

      for (auto m: adjacency[img]) {
        Image *other;
        Point2i offset;

        if (m->image_1 == img) {
          other = m->image_2;
          offset = Point2i(m->t_x, m->t_y);
        } else {
          other = m->image_1;
          offset = Point2i(-m->t_x, -m->t_y);
        }

        // If not yet assigned
        if (other->regInfo->component_membership != componentIndex) {
          other->regInfo->component_membership = componentIndex;
          other->regInfo->absoluteCoords = img->regInfo->absoluteCoords - offset;
          other->regInfo->matchedTo = img->index;
          other->regInfo->relativeCoords = offset;

          members.insert(other);
          q.push(other);
        }
      }
    }

    std::vector membersForRebuild(members.begin(), members.end());

    for (auto m: matches) {
      ig.addEdge(m->image_1->index, m->image_2->index, ImageGraph::EdgeKind::ORB);
    }
    for (int ii = 0; ii < extraMatches.size(); ++ii) {
      auto [img1,img2,kp1,kp2] = extraMatches[ii];
      ig.addEdge(img1->index, img2->index, ImageGraph::EdgeKind::SIFT);
    }


    for (auto img: members) {
      ig.setMember(img->index, true);
    }

    auto graphConnectivityResult = ig.computeMinPromotionsToConnectMembersPreferORB();
    if (!graphConnectivityResult.success) {
      std::cout << "component " << componentIndex << " failed to connect graph" << std::endl;
      return;
    }

    if (!graphConnectivityResult.promoted_nodes.empty()) {
      std::cout << "Component " << componentIndex << " promoting additional " << graphConnectivityResult.promoted_nodes.
          size() <<
          " frames in BA" << std::endl;
      //important to check if empty or get_image_ref returns every image known to StreamCam
      for (auto img: parent->get_image_ref(graphConnectivityResult.promoted_nodes)) {
        ig.setMember(img->index, true);
        members.insert(img);
      }
    }
    auto memberOverlaps = calculate_member_overlaps(std::vector(members.begin(), members.end()));
    auto start1 = std::chrono::high_resolution_clock::now();
    ImageGraph::PromoteMembersForOverlapConnectivityShortestHop(members, memberOverlaps, std::vector(ftg->storedMatches.begin(),ftg->storedMatches.end()));
    auto t4 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start1).count();


    for (auto m: matches) {
      if (members.find(m->image_1) != members.end() && members.find(m->image_2) != members.end()) {
        ++m->image_1->matchCount;
        ++m->image_2->matchCount;
        for (int i = 0; i < m->good_matches.size(); ++i) {
          if (m->inliers[i]) {
            ftg->process_match(m->image_1->index, m->image_2->index, m->good_matches[i]);
          }
        }
      }
    }


    for (auto &img: members) {
      img->keypointsImageSpace.resize(img->keypoints.size());
      for (int i = 0; i < img->keypoints.size(); ++i) {
        img->keypointsImageSpace[i].pt = img->keypoints[i].pt / parent->scale_factor;
      }
      img->regInfo->wasAligned = true;
    }

    if (graphConnectivityResult.used_sift) {
      std::cout << "Component " << componentIndex << " using sift in BA" << std::endl;
      return;
    }

    //GENERATE TRACKS AND RUN
    std::vector memberImgs(members.begin(), members.end());

    auto tracks = ftg->generateCurrentTracks(memberImgs);
    BundleAdjustmentIntegrator::run_coopers_planar_ba_edge_list(tracks, memberImgs, 2 * memberImgs.size() + 200);

    rebuild();

    auto t3 = std::chrono::duration_cast<std::chrono::milliseconds>
        (std::chrono::high_resolution_clock::now() - start).count();
    std::cout << "total align time comp " << componentIndex << ": " << t3 << std::endl;
  }

  void CompositeVoronoi::rebuild_and_initialize_SAM() {
    //currently SAM only works for one component at a time. Ensure this is a single resolution composite
    assert(componentIndex == 0);

    self_reset();
    cudaSetDevice(parent->compositorCudaDevice);

    //do this first so we have root and max offset determined ahead of time
    for (auto &img: contributingImages) {
      Point2f absC(img->absoluteCoords.x, img->absoluteCoords.y);
      std::vector<Point2i> face;
      if (add_point_to_delaunay_triangulation(absC, img, face, true, false) >= 0) {
        max_offset.x = max(max_offset.x, img->absoluteCoords.x + img->width);
        max_offset.y = max(max_offset.y, img->absoluteCoords.y + img->height);
        root_offset.x = min(root_offset.x, img->absoluteCoords.x);
        root_offset.y = min(root_offset.y, img->absoluteCoords.y);
      }
    }

    int interval = (parent->SAMTileSize / parent->tileSize);

    auto ul = imagePyramid->level[0]->getIJ(Point2f(root_offset.x, root_offset.y));
    auto lr = imagePyramid->level[0]->getIJ(Point2f(max_offset.x, max_offset.y));

    int id = 0;
    int yTileCount = 0;
    auto accessSAM = parent->as;

    for (int y = ul.y; y <= lr.y; ++y) {
      if ((yTileCount - 1) % (interval - 1) == 0) {
        int xTileCount = 0;

        for (int x = ul.x; x <= lr.x; ++x) {
          if ((xTileCount - 1) % (interval - 1) == 0) {
            auto st = new SAMTile(id, {x - 1, y - 1}, accessSAM.get(), 0, parent->SAMTileSize);

            accessSAM->tiles.push_back(st);
            ++id;

            //add links to neighbors
            if (xTileCount > 0) {
              auto brotherX = accessSAM->tiles[accessSAM->get_tile_id({x - interval + 1, y}, componentIndex)];
              std::vector<Point2i> temp;
              for (int yy = 0; yy < interval; ++yy) {
                temp.emplace_back(x - 1, y - 1 + yy);
              }
              accessSAM->tiles.back()->neighbors.emplace_back(brotherX, temp);
              brotherX->neighbors.emplace_back(accessSAM->tiles.back(), temp);
            }
            if (yTileCount > 0) {
              auto brotherY = accessSAM->tiles[accessSAM->get_tile_id({x, y - interval + 1}, componentIndex)];
              std::vector<Point2i> temp;
              for (int xx = 0; xx < interval; ++xx) {
                temp.emplace_back(x - 1 + xx, y - 1);
              }
              accessSAM->tiles.back()->neighbors.emplace_back(brotherY, temp);
              brotherY->neighbors.emplace_back(accessSAM->tiles.back(), temp);
            }

            //choose image
            Rect tileRect(st->location.x * parent->tileSize, st->location.y * parent->tileSize, parent->SAMTileSize,
                          parent->SAMTileSize);
            for (auto &brother: st->neighbors) {
              if (brother.first->img) {
                int coverage = pixels_overlapping_between(brother.first->img, tileRect);
                if (coverage == parent->SAMTileSize * parent->SAMTileSize) {
                  //take this image as st's image
                  st->img = brother.first->img;
                  st->imgIndex = st->img->index;
                }
              }
            }

            if (!st->img) {
              int bestCoverage = 0;
              for (auto &img: contributingImages) {
                int val = pixels_overlapping_between(img, tileRect);
                if (val > bestCoverage) {
                  bestCoverage = val;
                  st->img = img;
                  st->imgIndex = img->index;
                }
                if (bestCoverage == parent->SAMTileSize * parent->SAMTileSize) { break; }
              }
            }
          }
          ++xTileCount;
        }
      }
      ++yTileCount;
    }

    auto tempTiles = accessSAM->tiles;
    std::sort(tempTiles.begin(), tempTiles.end(),
              [](const SAMTile *a, const SAMTile *b) {
                return a->imgIndex < b->imgIndex;
              });

    auto currentInd = tempTiles[0]->imgIndex;
    for (auto &samTile: tempTiles) {
      if (!samTile->img) { continue; }

      samTile->valid = true;
      auto img = samTile->img;

      //check if we need to load a new image for the next group of SAM tiles
      if (samTile->imgIndex != currentInd) {
        //wait for buffer to be on GPU
        {
          std::unique_lock<std::mutex> lock(img->cudaBufferMutex);
          img->cudaBufferConVar.wait(lock, [&] { return img->cudaBufferReady; });
        }

        //prepare 3 channel image
        cuda::GpuMat image_Mat(imageSize, CV_8U, img->get_raw_cuda());
        cuda::cvtColor(image_Mat, threeChannelPrealGPU, COLOR_BayerBG2BGR);

        img->free_memory_cuda();

        if (componentMagLabel != 0) {
          //flatfield correct
          ff_correct_and_brighten();
        }

        //add alpha channel
        cuda::split(threeChannelPrealGPU, channelsGPU);
        if (componentMagLabel == Image::_2X) {
          for (auto &channel: channelsGPU) {
            cuda::multiply(channel, circleMaskGPU, channel);
          }
          channelsGPU.push_back(circleMaskGPU255);
        } else {
          channelsGPU.push_back(rectMaskGPU);
        }
        cuda::merge(channelsGPU, fourChannelPrealGPU);

        currentInd = samTile->imgIndex;
      }


      //populate SAM gpu mat with data from 3channel preal
      Rect tileRect(samTile->location.x * parent->tileSize, samTile->location.y * parent->tileSize,
                    parent->SAMTileSize, parent->SAMTileSize);
      Rect imageRect(img->absoluteCoords.x, img->absoluteCoords.y, img->width, img->height);
      Rect roi = tileRect & imageRect;

      Rect imageRoi = roi;
      imageRoi.x -= imageRect.x;
      imageRoi.y -= imageRect.y;

      Rect tileRoi = roi;
      tileRoi.x -= tileRect.x;
      tileRoi.y -= tileRect.y;

      fourChannelPrealGPU(imageRoi).copyTo(samTile->noncontiguousWrapper(tileRoi));
      samTile->noncontiguousWrapper.download(samTile->ncwStoreLocal);

      samTile->make_raw_buffer();

      std::vector<Point2i> retileIndices(9);
      for (int xx = 0; xx < 4; ++xx) {
        for (int yy = 0; yy < 4; ++yy) {
          Point2i sublocation(xx, yy);
          Point2i tileID(samTile->location.x + xx, samTile->location.y + yy);
          samTile->componentTiles.emplace_back(sublocation, tileID);
          if (xx < 3 && yy < 3) {
            retileIndices[3 * xx + yy] = tileID;
          }
        }
      }
      imagePyramid->insertTilesAtBase(fourChannelPrealGPU, rectMaskGPU, imageRect, retileIndices);
    }
  }


  void CompositeVoronoi::rebuild() {
    if (contributingFrames.size() == 1) {
      return;
    }
    self_reset();

    //do this for all images first so we pull final voronoi face on reconstruct
    for (auto &img: contributingFrames) {
      Point2f absC(img->regInfo->absoluteCoords.x, img->regInfo->absoluteCoords.y);
      std::vector<Point2i> face;
      if (add_point_to_delaunay_triangulation(absC, img, face, true, false) >= 0) {
        max_offset.x = max(max_offset.x, img->regInfo->absoluteCoords.x + img->width);
        max_offset.y = max(max_offset.y, img->regInfo->absoluteCoords.y + img->height);
        root_offset.x = min(root_offset.x, img->regInfo->absoluteCoords.x);
        root_offset.y = min(root_offset.y, img->regInfo->absoluteCoords.y);
      }
    }

    for (auto &img: contributingFrames) {
      //get voronoi facets for only this face
      std::vector<std::vector<Point2f> > facets;
      std::vector<Point2f> centers;
      std::vector<Point2i> face;

      int vertexId = -1;
      for (auto element: delaunayMembers) {
        if (element.second == img->index) {
          vertexId = element.first;
          break;
        }
      }
      assert(vertexId != -1);
      subdiv.getVoronoiFacetList({vertexId}, facets, centers);
      img->load_raw_from_disk(false);


      //shift and recast
      for (auto &ii: facets[0]) {
        //we have pulled only one face so facets has only 1 element
        ii.x -= centers[0].x;
        ii.x += imageSize.width / 2;
        ii.y -= centers[0].y;
        ii.y += imageSize.height / 2;
        face.push_back((Point2i) ii);
      }
      clean_face(face);

      fillConvexPoly(polyMaskOutput, face, cv::Scalar(255));
      // polyMaskGPU.upload(polyMaskOutput);

      auto ri = img->regInfo;

      std::vector<Point2i> effectedTiles;
      std::vector<Point2i> effectedTilesNoMask;
      bool noMask = false;

      //calculate region of pyramid for data placement
      auto imageBox = cv::Rect_<float>(ri->absoluteCoords.x, ri->absoluteCoords.y, img->width,
                                       img->height);

      if (componentMagLabel == Image::_2X) {
        calculate_effected_tiles_round(face, effectedTiles, ri->absoluteCoords);
      } else {
        noMask = true;
        calculate_effected_tiles(face, effectedTiles, ri->absoluteCoords, &effectedTilesNoMask);

      }

      auto tiles = effectedTilesNoMask;
      tiles.insert(tiles.end(),effectedTiles.begin(),effectedTiles.end());
      prepare_4CPA(img, tiles);
      if (!effectedTilesNoMask.empty()) {
        imagePyramid->insertTilesAtBase(fourChannelPreallocated,rectMask,imageBox,effectedTilesNoMask);
      }
      imagePyramid->insertTilesAtBase(fourChannelPreallocated, polyMaskOutput, imageBox, effectedTiles);
      polyMaskOutput.setTo(Scalar(0));
      parent->notify_observers();

      img->free_memory_RAW();
    }
    //needsAlignment = false;
  }

  int CompositeVoronoi::pixels_overlapping_between(Image *_img, Rect _rect) {
    Rect imageRect(_img->regInfo->absoluteCoords.x, _img->regInfo->absoluteCoords.y, _img->width, _img->height);
    auto overlapRect = imageRect & _rect;

    if (overlapRect.area() == 0 || componentMagLabel != Image::_2X) {
      return overlapRect.area();
    }

    overlapRect.x -= _img->regInfo->absoluteCoords.x;
    overlapRect.y -= _img->regInfo->absoluteCoords.y;
    return cuda::countNonZero(circleMaskGPU(overlapRect));
  }


  SiftData Composite::GPU_extract_SIFT(cuda::GpuMat &_img, int _numPts) {
    SiftData siftData;
    try {
      if (_img.channels() == 1) {
        cuda::cvtColor(_img, gry, COLOR_BayerBG2GRAY);
      } else if (_img.channels() == 3) {
        cuda::cvtColor(_img, gry, COLOR_BGR2GRAY);
      } else {
        throw std::runtime_error("Unsupported image format in GPU_extract_SIFT");
      }

      if (componentMagLabel == Image::_2X) {
        cuda::multiply(circleMaskGPU, gry, gry);
      }

      gry.convertTo(gry2,CV_32FC1);

      CudaImage cImgGry;
      cImgGry.Allocate(imageSize.width, imageSize.height, gry2.step / sizeof(float), false,
                       reinterpret_cast<float *>(gry2.data), nullptr);


      InitSiftData(siftData, 100000, true, true);
      catch_ExtractSift(siftData, cImgGry, 5, 0.0f, 0.4f, 0.1f, false);
    } catch (cv::Exception &e) {
      int k = 0;
    }
    return siftData;

    //int k = 0;
  }


  std::vector<std::pair<Image *, Image *> > CompositeVoronoi::calculate_new_overlaps() {
    std::vector<std::pair<Image *, Image *> > newOverlaps;
    double radSq = pow(0.8 * parent->scope_radius, 2);

    //overlaps within component
    for (int i = 0; i < contributingRegInfos.size() - 1; ++i) {
      if (componentMagLabel == Image::_2X) {
        if (pow(contributingRegInfos[i]->absoluteCoords.x - contributingRegInfos.back()->absoluteCoords.x, 2) +
            pow(contributingRegInfos[i]->absoluteCoords.y - contributingRegInfos.back()->absoluteCoords.y, 2) < radSq) {
          throw std::runtime_error("this logic path is no longer functional");
          //newOverlaps.push_back({contributingImages[i], contributingImages.back()});
        }
      } else {
        if (abs(contributingRegInfos[i]->absoluteCoords.x - contributingRegInfos.back()->absoluteCoords.x) < 0.7 *
            imageSize.
            width &&
            abs(contributingRegInfos[i]->absoluteCoords.y - contributingRegInfos.back()->absoluteCoords.y) < 0.7 *
            imageSize.
            height) {
          throw std::runtime_error("this logic path is no longer functional");
          //newOverlaps.emplace_back(contributingImages[i], contributingImages.back());
        }
      }
    }
    /*
        //overlaps between this and other components
        for (auto comp: parent->composites) {

          if (comp != this) {
            for (auto di: comp->delaunayImages) {
              if (delaunayRegInfos.back()->root) {
                newOverlaps.emplace_back(di, delaunayImages.back());
              } else {
                assert(imagePyramid->scale != 0);
                //calculate my position in base space
                auto myBaseAbC = delaunayRegInfos.back()->get_AbC_relative_from_local(0);
                auto theirBaseAbC = di->regInfo->get_AbC_relative_from_local(0);

                double myScale = imagePyramid->scale;
                double theirScale = comp->imagePyramid->scale;

                myBaseAbC.x += myScale * 0.5 * parent->image_width;
                myBaseAbC.y += myScale * 0.5 * parent->image_height;

                theirBaseAbC.x += theirScale * 0.5 * parent->image_width;
                theirBaseAbC.y += theirScale * 0.5 * parent->image_height;

                double allowableDiffX = theirScale + myScale * 0.5 * parent->image_width;
                double allowableDiffY = theirScale + myScale * 0.5 * parent->image_height;

                auto centerDiff = theirBaseAbC - myBaseAbC;
                centerDiff.x = abs(centerDiff.x);
                centerDiff.y = abs(centerDiff.y);

                if (centerDiff.x <= 0.7 * allowableDiffX && centerDiff.y <= 0.7 * allowableDiffY) {
                  newOverlaps.emplace_back(di, delaunayImages.back());
                }
              }
            }
          }
        }
    */
    std::reverse(newOverlaps.begin(), newOverlaps.end());
    return newOverlaps;
  }

  void CompositeVoronoi::ff_correct_and_brighten() {
    if (componentMagLabel != 0) {
      //flatfield correct
      threeChannelPrealGPU.convertTo(convertHoldingGPU, CV_32F, parent->cvCompositeStream);
      cuda::divide(convertHoldingGPU, ffGPU, convertHoldingGPU, 1, CV_32F, parent->cvCompositeStream);
      //brighten
      cuda::pow(convertHoldingGPU, 1.1, convertHoldingGPU, parent->cvCompositeStream);
      convertHoldingGPU.convertTo(threeChannelPrealGPU, CV_8UC3, parent->cvCompositeStream);
    }
  }

#endif
}
