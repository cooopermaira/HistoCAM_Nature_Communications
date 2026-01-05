//
// Created by cooper on 5/9/25.
//
#include "pathCam.h"

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
    Mat x_row(1, image_size.width, CV_16S);
    Mat y_col(image_size.height, 1, CV_16S);

    for (int i = 0; i < image_size.width; i++) {
      x_row.at<short>(i) = i;
    }
    for (int i = 0; i < image_size.height; i++) {
      y_col.at<short>(i) = i;
    }

    Mat X, Y;
    repeat(x_row, image_size.height, 1, X);
    repeat(y_col, 1, image_size.width, Y);

    meshGridX.upload(X);
    meshGridY.upload(Y);

    diffGPU = cuda::GpuMat(image_size, CV_32S);
    xp1 = cuda::GpuMat(image_size, CV_32F);
    xp2 = cuda::GpuMat(image_size, CV_32F);
    binaryCompare = cuda::GpuMat(image_size, CV_8U);

    polyMaskGPU = cuda::GpuMat(image_size, CV_8U);
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

  void adjust_roi_for_debayer(Rect &roi_) {
    if (roi_.x % 2 > 0) {
      --roi_.x;
      ++roi_.width;
    }
    if (roi_.y % 2 > 0) {
      --roi_.y;
      ++roi_.height;
    }
    if (roi_.width % 2 > 0) {
      ++roi_.width;
    }
    if (roi_.height % 2 > 0) {
      ++roi_.height;
    }
  }

  bool Composite::prepare_4CPA(Image *img, const std::vector<Point2i> &affectedTiles, const bool forceFullImage) {
    if (affectedTiles.size() < 100 && !forceFullImage) {
      bool ans = false;

      img->regInfo->accessMutex->lock();
      auto AbC = img->regInfo->absoluteCoords;
      img->regInfo->accessMutex->unlock();

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

        threeChannelPrealGPU = cuda::GpuMat(imageSize,CV_8UC3, threeChannelPreallocated.data);
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
      //brighten (now done by scaling flatfield image instead
      //cuda::pow(convertHoldingGPU(roi_), 1.05, convertHoldingGPU(roi_),parent->cvCompositeStream);
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


  void CompositeVoronoi::GPU_add_images_no_composite(std::vector<RegInfo *> _newInfo, bool _force_add) {
    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<unsigned long> indexes;
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
        rawMat = cuda::GpuMat(image_size, CV_8U, images[i]->get_raw_cuda());
        cuda::cvtColor(rawMat, threeChannelPrealGPU, COLOR_BayerBG2BGR, 0, parent->cvCompositeStream);
      } else {
        Mat rawMat;
        rawMat = Mat(image_size, CV_8U, images[i]->get_Raw());
        cvtColor(rawMat, threeChannelPreallocated, COLOR_BayerBG2BGR);
        threeChannelPrealGPU = cuda::GpuMat(image_size,CV_8UC3, threeChannelPreallocated.data);
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
                                Image::get_label(componentMagLabel));

      parent->notify_observers();
    }
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
            auto st = new SAMTile(id, {x - 1, y - 1}, accessSAM, 0, parent->SAMTileSize);

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
        cuda::GpuMat image_Mat(image_size, CV_8U, img->get_raw_cuda());
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
    if (contributingImages.size() == 1) {
      //return;
    }
    self_reset();

    //do this for all images first so we pull final voronoi face on reconstruct
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

    for (auto &img: contributingImages) {
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


      //shift and recast
      for (auto &ii: facets[0]) {
        //we have pulled only one face so facets has only 1 element
        ii.x -= centers[0].x;
        ii.x += image_size.width / 2;
        ii.y -= centers[0].y;
        ii.y += image_size.height / 2;
        face.push_back((Point2i) ii);
      }
      clean_face(face);

      fillConvexPoly(polyMaskOutput, face, cv::Scalar(255));
      polyMaskGPU.upload(polyMaskOutput);

      //wait for buffer to be on gpu
      {
        std::unique_lock<std::mutex> lock(img->cudaBufferMutex);
        img->cudaBufferConVar.wait(lock, [&] { return img->cudaBufferReady; });
      }

      //debayer image on gpu
      cuda::GpuMat image_Mat(image_size, CV_8U, img->get_raw_cuda());
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

      //calculate effected tiles
      std::vector<Point2i> effectedTiles;
      std::vector<Point2i> effectedTilesNoMask;

      //calculate region of pyramid for data placement
      auto imageBox = cv::Rect_<float>(img->absoluteCoords.x, img->absoluteCoords.y, img->width,
                                       img->height);

      if (componentMagLabel == Image::_2X) {
        calculate_effected_tiles_round(face, effectedTiles, img->absoluteCoords);
        cuda::multiply(polyMaskGPU, circleMaskGPU, polyMaskGPU);
      } else {
        calculate_effected_tiles(face, effectedTiles, img->absoluteCoords, &effectedTilesNoMask);
        imagePyramid->insertTilesAtBase(fourChannelPrealGPU, rectMaskGPU, imageBox, effectedTilesNoMask);
      }

      imagePyramid->insertTilesAtBase(fourChannelPrealGPU, polyMaskGPU, imageBox, effectedTiles);

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
      polyMaskOutput.setTo(Scalar(0));
      parent->notify_observers();
    }
    //needsAlignment = false;
  }

  int CompositeVoronoi::pixels_overlapping_between(Image *_img, Rect _rect) {
    Rect imageRect(_img->absoluteCoords.x, _img->absoluteCoords.y, _img->width, _img->height);
    auto overlapRect = imageRect & _rect;

    if (overlapRect.area() == 0 || componentMagLabel != Image::_2X) {
      return overlapRect.area();
    }

    overlapRect.x -= _img->absoluteCoords.x;
    overlapRect.y -= _img->absoluteCoords.y;
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
            image_size.
            width &&
            abs(contributingRegInfos[i]->absoluteCoords.y - contributingRegInfos.back()->absoluteCoords.y) < 0.7 *
            image_size.
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
