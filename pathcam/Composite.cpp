//
//  Composite.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/24/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam {
  CompositeVoronoi::CompositeVoronoi(StreamCam *parent, cv::Size image_size,
                                     unsigned int component_index) : Composite(
                                                                       parent), componentIndex(component_index),
                                                                     wakeEvent(true), image_size(image_size) {
    minPixelDistanceBetweenFrames = 200;

    imagePyramid.reset(new MRTiledImage(parent));
    std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(
      imagePyramid, parent->tileSize, parent->tileSize,
      0);
    imagePyramid->level.push_back(current);
    parent->MRimage->add(imagePyramid);

    subdiv_Bbox = Bbox(-50000, -50000, 50000, 50000);
    subdiv.initDelaunay(subdiv_Bbox.as_cvRect());

    circleMask = cv::Mat::zeros(image_size, CV_8U);
    cv::circle(circleMask, cv::Point(image_size.width / 2, image_size.height / 2), parent->scope_radius,
               cv::Scalar(1),
               -1);
    rectMask = Mat(image_size, CV_8U, cv::Scalar(255));

#ifdef HAVE_OPENCV_CUDAARITHM
    circleMaskGPU.upload(circleMask);
    circleMaskGPU255 = cuda::GpuMat(image_size,CV_8U);
    cuda::multiply(circleMaskGPU,Scalar(255),circleMaskGPU255);
    rectMaskGPU.upload(rectMask);
    channelsGPU.resize(4);
    threeChannelPrealGPU = cuda::GpuMat(image_size, CV_8UC3);
    fourChannelPreallocated = Mat::zeros(image_size, CV_8UC4);
    //make_meshgrid();
#endif

    channels.resize(2);

    imageBoundsAsPolygon.resize(4);
    reset_image_as_polygon();

    polyMaskOutput = Mat::zeros(image_size, CV_8U);
    freshMask = polyMaskOutput.clone();


    //first added image will be at (0,0), this value guarantees it is accepted
    lastAcceptedImageIndex = 999999999;
  }


  void CompositeVoronoi::update(std::vector<RegInfo *> new_info, bool _force_add) {
    if (new_info.empty()){return;}

    update_mutex->lock();

    if (new_info.size() > 1) {
      // this shuffle is very important for reducing image count in the DT.
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(new_info), std::end(new_info), rng);
    }

    update_Bbox_no_composite(new_info);
    expand_subdiv(new_info);
#ifdef HAVE_OPENCV_CUDAARITHM
    GPU_add_images_no_composite(new_info, _force_add);


#else
    add_images_no_composite(new_info, _force_add);
#endif
    update_mutex->unlock();
  }

  void CompositeVoronoi::store_new_info(pathCam::RegInfo *_new_info) {
    storedNewInfo = _new_info;
  }

  void CompositeVoronoi::update_from_stored_info() {
    update({storedNewInfo});
  }

  void CompositeVoronoi::check_set_render_info() {
    if (imagePyramid->scale == 0) {
      double scale;
      Point2f offset;
      if (parent->get_scale_and_offset(componentIndex, scale, offset)) {
        imagePyramid->set_scale(scale);
        imagePyramid->set_offset(offset);

        storedNewInfo->set_abc(Vec2(0, 0), componentIndex, true);
        deduce_label();
      }
    }
  }

  void CompositeVoronoi::set_offset(const Point2f &_offset) const {
    imagePyramid->set_offset(_offset);
  }


  void CompositeVoronoi::set_scale(double _scale) {
    imagePyramid->set_scale(_scale);
    deduce_label();
    imagePyramid->set_mag_label(componentMagLabel);
    parent->MRimage->sort_by_scale();
  }


  void CompositeVoronoi::deduce_label() {
    if (componentIndex == 0) {
      componentMagLabel = parent->initialLabel;
      get_flatfield();
      return;
    }

    if (imagePyramid->scale == 0) {
      componentMagLabel = 0;
      return;
    }

    if (parent->composites[0]->componentMagLabel == 0) {
      return;
    }

    double initialComponentTrueScale = parent->labelScales[parent->composites[0]->componentMagLabel];
    double selfTrueScale = imagePyramid->scale * initialComponentTrueScale;

    double closest = parent->labelScales[0];
    double minDiff = std::abs(selfTrueScale - parent->labelScales[0]);

    // Find the closest value
    for (int i = 1; i < parent->labelScales.size(); i++) {
      double diff = std::abs(selfTrueScale - parent->labelScales[i]);
      if (diff < minDiff) {
        minDiff = diff;
        closest = parent->labelScales[i];
      } else {
        componentMagLabel = i - 1;
        get_flatfield();
        return;
      }
    }
    componentMagLabel = parent->labelScales.size() - 1;
    get_flatfield();

  }

  void CompositeVoronoi::get_flatfield() {
    set_candidate_scale_ratios();

    std::string filename = parent->get_flatfield(componentMagLabel);
    size_t nBytes = parent->image_height * parent->image_width;
    char *buffer = new char[nBytes];
    char *bufferCuda;
    std::ifstream stream;
    stream.open(filename, std::ios::binary);
    stream.read(buffer, nBytes);

#ifdef HAVE_OPENCV_CUDAARITHM
    CHECK_CUDA(cudaMalloc(&bufferCuda, nBytes));
    CHECK_CUDA(cudaMemcpy(bufferCuda, buffer, nBytes, cudaMemcpyHostToDevice));

    ffGPU = cuda::GpuMat(Size(parent->image_width, parent->image_height), CV_8U, bufferCuda);

    cuda::cvtColor(ffGPU, ffGPU, COLOR_BayerBG2BGR);
    ffGPU.convertTo(ffGPU,CV_32F);
    double scale = 1 / 170.0;
    cuda::multiply(ffGPU, Scalar(scale, scale, scale), ffGPU);
    // if (componentMagLabel == 2) {
    //   Mat temp;
    //   ffGPU.download(temp);
    //   imwrite("/media/max/Data/4xff.png", temp);
    //   int k = 0;
    // }
#else
    ff = Mat(Size(6464, 4852), CV_8U, bufferCuda, Mat::AUTO_STEP);
    cvtColor(ff,ff,COLOR_BayerBG2BGR);
    ff.convertTo(ff,CV_32F);
    ff *= 1/170.0;
#endif
  }

  void CompositeVoronoi::set_candidate_scale_ratios() {
    switch (componentMagLabel) {
      case Image::_2X:
        candidateScaleRatios = {1.0, 2.0, 5.0, 10.0, 20.0};
        break;
      case Image::_4X:
        candidateScaleRatios = {0.5, 1.0, 2.5, 5.0, 10.0};
        break;
      case Image::_10X:
        candidateScaleRatios = {0.2, 0.4, 1.0, 2.0, 4.0};
        break;
      case Image::_20X:
        candidateScaleRatios = {0.1, 0.2, 0.5, 1.0, 2.0};
        break;
      case Image::_40X:
        candidateScaleRatios = {0.05, 0.1, 0.25, 0.5, 1.0};
        break;
      default:
        throw std::runtime_error("unknown component mag label");
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
    root_offset = Vec2(0, 0);
    max_offset = Vec2(0, 0);
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
      img->absoluteCoords = ni->absoluteCoords;
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
    _image->correct_registration({});
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
      _image->correct_registration(adjacentVerts);
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
      ii.x += image_size.width / 2;
      ii.y -= centers[0].y;
      ii.y += image_size.height / 2;
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
    _image->absoluteCoords = Vec2(point.x, point.y);
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
      ii.x += image_size.width / 2;
      ii.y -= centers[0].y;
      ii.y += image_size.height / 2;
      _face.push_back((Point2i) ii);
    }

    clean_face(_face);
    //build polygon mask for new point
#ifdef HAVE_OPENCV_CUDAARITHM
    // coopers_GPU_vectorized_convex_mask_maker(_face);
    // //cuda::multiply(polyMaskGPU,circleMaskGPU,polyMaskGPU);
    // Mat temp;
    // polyMaskGPU.download(temp);

    if (_drawMask) {
      polyMaskOutput.setTo(Scalar(0));
      fillConvexPoly(polyMaskOutput, _face, cv::Scalar(255));
    }
    // imwrite("/media/max/Data/fcp.png", polyMaskOutput);
    // imwrite("/media/max/Data/ccmm.png", temp);

#else
        cv::fillConvexPoly(polyMaskOutput, _face, cv::Scalar(255));
#endif

    //test for exclusion of frame via rollback
    int nonzeroMin;
    if (componentMagLabel == Image::_2X) {
      polyMaskOutput = polyMaskOutput.mul(circleMask);
      nonzeroMin = parent->scope_radius * parent->scope_radius * 3.14 * 0.00;
    } else {
      nonzeroMin = _image->width * _image->height * 0.1;
    }

    if (!_forceAdd && countNonZero(polyMaskOutput) <= nonzeroMin) {
      //contributing less than x% of its pixels, revert and don't bother loading from disk
      subdiv = tempSubdiv;
      _image->free_memory_RAW();
      memberImages.push_back({_image, false});
      return -1;
    }

    _image->absoluteCoords = Vec2(_point.x, _point.y);
    memberImages.push_back({_image, true});
    delaunayMembers.insert({vertxId, _image->index});
    return vertxId;
  }


  void CompositeVoronoi::add_images_no_composite(std::vector<RegInfo *> new_info, bool _force_add) {
    auto start = std::chrono::high_resolution_clock::now();

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<unsigned long> indexes;
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
      Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
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
                                Image::get_label(componentMagLabel));

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
                                                  Vec2 absCoord, std::vector<Point2i> *additionalResult) {
    std::vector<Point2i> tileIndices;
    std::map<int, std::vector<float> > tilesByColumn;

    //get the tile column of the left and right edges
    int columnBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).x;
    int columnBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x + image_size.width, absCoord.y)).x;

    //get the tile row top and bottom edges of the image frame
    long rowBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).y;
    long rowBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x, absCoord.y + image_size.height)).y;

    //get the tile row top and bottom as if the image was square. This is used to prevent chevrons
    auto dimensionDifference = parent->image_width - parent->image_height;
    long rowBoundFalseLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y - dimensionDifference / 2))
        .y;
    long rowBoundFalseHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x, absCoord.y + image_size.height + dimensionDifference / 2)).y;

    float yPixelBoundLow = float(rowBoundFalseLow) * float(imagePyramid->tile_size);
    float yPixelBoundHigh = float(rowBoundFalseHigh) * float(imagePyramid->tile_size);

    //for each edge of the voronoi mask
    for (int ii = 0; ii < maskAsPolygon.size() - 1; ii++) {
      //if we are at the last point, make the next point the first point (this makes the last edge)
      //int ii2 = (ii + 1) == maskAsPolygon.size() ? 0 : ii + 1;
      int ii2 = ii + 1;

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

        firstPoint_y = std::max((double) firstPoint_y, absCoord.y);
        lastPoint_y = std::min((double) lastPoint_y, absCoord.y + parent->image_height);

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

            if (abs(absCoord.x + image_size.width / 2 - x) < image_size.width / 2
                && abs(absCoord.y + image_size.height / 2 - y) < image_size.height / 2) {
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

  void
  CompositeVoronoi::calculate_effected_tiles_round(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result,
                                                   Vec2 absCoord) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Point2i> tileIndices;
    std::map<int, std::vector<float> > tilesByColumn;

    //get the tile column of the left and right edges of the image frame
    int columnBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).x;
    int columnBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x + image_size.width, absCoord.y)).x;

    //get the tile row the top and bottom edges of the image frame
    long rowBoundLow = imagePyramid->level[0]->getIJ(Point2f(absCoord.x, absCoord.y)).y;
    long rowBoundHigh = imagePyramid->level[0]->getIJ(
      Point2f(absCoord.x, absCoord.y + image_size.height)).y;

    float yPixelBoundLow = float(rowBoundLow) * float(imagePyramid->tile_size);
    float yPixelBoundHigh = float(rowBoundHigh) * float(imagePyramid->tile_size);

    //for each edge of the voronoi mask
    for (int ii = 0; ii < maskAsPolygon.size(); ii++) {
      //if we are at the last point, make the next point the first point (this makes the last edge)
      int ii2 = (ii + 1) == maskAsPolygon.size() ? 0 : ii + 1;

      //create an Point2f from the cv::Point2i
      auto p1 = Point2f(maskAsPolygon[ii].x + absCoord.x, maskAsPolygon[ii].y + absCoord.y);
      auto p2 = Point2f(maskAsPolygon[ii2].x + absCoord.x, maskAsPolygon[ii2].y + absCoord.y);

      //test for duplicate points that result from the voronoi calculation
      if (p1.x == p2.x && p1.y == p2.y) {
        continue;
      }

      //retreive the tiles these points fall within
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
      if (columnBoundHigh < xlow || columnBoundLow > xhigh) {
        continue; //no intersection with frame tiles
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

    int scopeRadSqr = pow(parent->scope_radius, 2);
    for (auto &[key, value]: tilesByColumn) {
      float firstPoint_y = *std::min_element(tilesByColumn[key].begin(), tilesByColumn[key].end());
      float lastPoint_y = *std::max_element(tilesByColumn[key].begin(), tilesByColumn[key].end());

      if (lastPoint_y < yPixelBoundLow || firstPoint_y > yPixelBoundHigh) {
        continue;
      }

      int lastTile = imagePyramid->level[0]->getIJ(
        Point2f(key * imagePyramid->level[0]->getTileSize(), lastPoint_y)).y;
      int firstTile = imagePyramid->level[0]->getIJ(
        Point2f(key * imagePyramid->level[0]->getTileSize(), firstPoint_y)).y;

      for (int ii = firstTile; ii <= lastTile; ii++) {
        if (ii <= rowBoundHigh && ii >= rowBoundLow) {
          //make sure at least one corner is closer than scope radius from center
          for (int i = 0; i < 4; i++) {
            int x = key * imagePyramid->tile_size + (i % 2 == 0 ? imagePyramid->tile_size : 0);
            int y = ii * imagePyramid->tile_size + (i % 3 == 0 ? imagePyramid->tile_size : 0);
            auto dist = pow(absCoord.x + image_size.width / 2 - x, 2) + pow(
                          absCoord.y + image_size.height / 2 - y, 2);
            if (dist < scopeRadSqr) {
              result.push_back(Point2i(key, ii));
              if (key < minTilex) { minTilex = key; }
              if (key > maxTilex) { maxTilex = key; }
              if (ii < minTiley) { minTiley = ii; }
              if (ii > maxTiley) { maxTiley = ii; }
              break;
            }
          }
        }
      }
    }
  }


  void CompositeVoronoi::debug_write_contribution_on_grid(std::string name, pathCam::Vec2 absCoord, cv::Mat &img,
                                                          cv::Mat &mask) {
    int k = int(absCoord.x);
    k = 512 - k % 512;
    k = abs(k);
    Mat debugmat = Mat::zeros(4852, 6464, CV_8UC4);
    img.copyTo(debugmat, mask);
    for (int m = k; m < debugmat.cols; m += 512) {
      cv::line(debugmat, cv::Point(m, 0), cv::Point(m, 4852), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(m + 1, 0), cv::Point(m + 1, 4852), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(m + 2, 0), cv::Point(m + 2, 4852), Scalar(0, 0, 255, 255));
    }
    k = int(absCoord.y);
    k = 512 - k % 512;

    k = abs(k);
    for (int m = k; m < debugmat.rows; m += 512) {
      cv::line(debugmat, cv::Point(0, m), cv::Point(6464, m), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(0, m + 1), cv::Point(6464, m + 1), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(0, m + 2), cv::Point(6464, m + 2), Scalar(0, 0, 255, 255));
    }
    imwrite(name, debugmat);
  }

  void CompositeVoronoi::debug_draw_voronoi_face(cv::Mat img, std::vector<Point2i> maskAsPolygon,
                                                 int line_thickness) {
    for (int ii = 0; ii < maskAsPolygon.size(); ii++) {
      int ii2 = (ii + 1) == maskAsPolygon.size() ? 0 : ii + 1;

      if (maskAsPolygon[ii].x == maskAsPolygon[ii2].x && maskAsPolygon[ii].y == maskAsPolygon[ii2].y) {
        continue;
      }

      cv::line(img, maskAsPolygon[ii], maskAsPolygon[ii2], Scalar(0, 0, 0, 255), line_thickness);
    }
  }

  void
  CompositeVoronoi::debug_draw_voronoi(cv::Mat &img, cv::Subdiv2D &subdiv, bool _drawPathInsteadOfFaces,
                                       bool _drawIntersect, Point2i _intrCenter) {
    std::vector<std::vector<cv::Point2f> > facets;
    std::vector<cv::Point2f> centers;
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

    auto endColor = Scalar(130, 255, 130, 255);
    auto midColor = Scalar(0, 255, 255, 255);
    auto startColor = Scalar(130, 130, 255, 255);

    if (_drawIntersect) {
      auto copyMatCirc = img.clone();
      auto copyMatPoly = img.clone();

      circle(copyMatCirc, _intrCenter, parent->scope_radius, Scalar(1, 1, 1, 1), -1);

      std::vector<cv::Point> poly;
      for (cv::Point2f p: facets.back()) {
        p.x -= root_offset.x;
        p.y -= root_offset.y;

        p.x += image_size.width / 2;
        p.y += image_size.height / 2;
        poly.push_back(p);
      }
      if (!poly.empty()) {
        cv::fillConvexPoly(copyMatPoly, poly, cv::Scalar(120, 120, 120, 255));
      }

      copyMatCirc = copyMatCirc.mul(copyMatPoly);
      copyMatCirc.copyTo(img);
    }

    for (double i = 0; i < facets.size(); i++) {
      std::vector<cv::Point> poly;
      for (cv::Point2f p: facets[i]) {
        p.x -= root_offset.x;
        p.y -= root_offset.y;

        p.x += image_size.width / 2;
        p.y += image_size.height / 2;
        poly.push_back(p);
      }
      centers[i].x -= root_offset.x;
      centers[i].y -= root_offset.y;

      centers[i].x += image_size.width / 2;
      centers[i].y += image_size.height / 2;

      if (!_drawPathInsteadOfFaces) {
        if (!poly.empty()) {
          cv::polylines(img, poly, true, cv::Scalar(0, 0, 0, 255), 40, cv::LINE_AA);
        }
        if (i == 0) {
          cv::circle(img, centers[i], 150, Scalar(0, 255, 0, 255), cv::FILLED, cv::LINE_AA);
        } else if (facets.size() > 1 && i == facets.size() - 1) {
          cv::circle(img, centers[i], 150, Scalar(0, 140, 255, 255), cv::FILLED, cv::LINE_AA);
        } else {
          cv::circle(img, centers[i], 150, cv::Scalar(255, 200, 0, 255), cv::FILLED, cv::LINE_AA);
        }
        cv::rectangle(img, cv::Point(10, 10), cv::Point(img.cols - 20, img.rows - 20), cv::Scalar(0, 0, 0, 255),
                      50);
      } else {
        if (i == 0) {
          circle(img, centers[0], 300, startColor, -1);
          continue;
        }
        double colorIndeptVar = (i / double(facets.size() / 2)); //spans from 0 to 2
        double colorShare1 = max(0.0, 1.0 - colorIndeptVar);
        //spans 1 to 0 at 50% of domain, stays 0 from 50% to end
        double colorShare3 = max(0.0, colorIndeptVar - 1.0);
        double colorShare2 = 1.0 - max(colorShare1, colorShare3);

        line(img, centers[i], centers[i - 1],
             colorShare1 * startColor + colorShare2 * midColor + colorShare3 * endColor, 150);
        if (i + 1 == facets.size()) {
          circle(img, centers.back(), 300, endColor, -1);
        }
      }
    }
  }


  void CompositeVoronoi::save_pyramid_as_image(std::string _fileName, bool _withGrid, bool _withGridAndIndexes,
                                               bool _withEffectedTiles, bool _outline,
                                               std::vector<Point2i> effectedTiles) {
    int lineThickness = 40;
    auto level = imagePyramid->level[0];

    //    auto ul = Point2i(minTilex, minTiley);
    //    auto lr = Point2i(maxTilex, maxTiley);
    auto ul = imagePyramid->level[0]->getIJ(Point2f(root_offset.x, root_offset.y));
    auto lr = imagePyramid->level[0]->getIJ(Point2f(max_offset.x, max_offset.y));

    int tile_size = level->getTileSize();


    int width = (lr.x + 1 - ul.x) * tile_size;
    int height = (lr.y + 1 - ul.y) * tile_size;

    width = std::abs(width);
    height = std::abs(height);

    Size pyramidSize = Size(width, height);
    Mat pyramidImage = Mat(pyramidSize, CV_8UC4);

    //we need to shift the x and y tiles so we aren't writing to negtive coordinates
    int x_offset = -ul.x;
    int y_offset = -ul.y;

    Mat tile;
    Mat greyBlend = Mat(Size(tile_size, tile_size), CV_8UC4, Scalar(120, 120, 120, 255));
    for (int x = ul.x; x <= lr.x; x++) {
      for (int y = ul.y; y <= lr.y; y++) {
        try {
          if (_withEffectedTiles) {
            if (imagePyramid->level[0]->tiles(x, y) == NULL) {
              tile = Mat(Size(tile_size, tile_size), CV_8UC4, Scalar(0, 0, 0, 0));
            } else {
#ifdef HAVE_OPENCV_CUDAARITHM
              level->getTile(x, y).image.download(tile);
#else
                            tile = level->getTile(x, y).clone();
#endif
            }
            auto searchForTile = Point2i(x, y);
            if (std::find(effectedTiles.begin(), effectedTiles.end(), Point2i(x, y)) != effectedTiles.
                end()) {
              tile = 0.5 * tile + 0.5 * greyBlend;
            }
          } else {
#ifdef HAVE_OPENCV_CUDAARITHM
            level->getTile(x, y).image.download(tile);
#else
                        tile = level->getTile(x, y).clone();
#endif
          }

          if (_withGrid) {
            cv::line(tile, cv::Point(0, 0), cv::Point(0, tile_size - 1), Scalar(0, 0, 0, 255),
                     lineThickness);
            cv::line(tile, cv::Point(0, tile_size - 1), cv::Point(tile_size - 1, tile_size - 1),
                     Scalar(0, 0, 0, 255), lineThickness);
            cv::line(tile, cv::Point(tile_size - 1, 0), cv::Point(tile_size - 1, tile_size - 1),
                     Scalar(0, 0, 0, 255), lineThickness);
            cv::line(tile, cv::Point(tile_size - 1, 0), cv::Point(0, 0), Scalar(0, 0, 0, 255),
                     lineThickness);
          }

          if (_withGridAndIndexes) {
            putText(tile, "(" + std::to_string(x + x_offset) + "," + std::to_string(y + y_offset) + ")",
                    Point(10, 50),
                    FONT_HERSHEY_PLAIN, 3, Scalar(0, 0, 0, 255), 5);
          }

          // Mat mask;
          // cv::extractChannel(tile, mask, 3);
          // double val;
          // int maglab = componentMagLabel;
          // maglab == 1
          //   ? val = 1.0
          //   : maglab == 2
          //       ? val = 0.5
          //       : maglab == 3
          //           ? val = 0.2
          //           : maglab == 4
          //               ? val = 0.1
          //               : val = 1;
          // double beta = (log2(1.0 / val) / 3.4) * 0.7 + 0.05;
          // auto greenShade = Mat(tile.rows, tile.cols, CV_8UC3,
          //                       Scalar(200 * beta, 150 * (1 - beta), 100 * beta));
          // std::vector<Mat> v = {greenShade, mask};
          // merge(v, greenShade);
          //
          // tile = beta * greenShade + (1 - beta) * tile;

          //imwrite(std::to_string(componentIndex) + "_" + std::to_string(x) + "_" + std::to_string(y) + ".png", tile);
          tile.copyTo(pyramidImage(Rect((x + x_offset) * tile.cols, (y + y_offset) * tile.rows, tile.cols,
                                        tile.rows)));
        } catch (const cv::Exception &e) {
          auto k = e.what();
          int kk = 0;
        }
      }
    }
    if (_outline) {
      rectangle(pyramidImage, Point2i(10, 10), Point2i(pyramidImage.cols - 20, pyramidImage.rows - 20),
                Scalar(0, 0, 0, 255), 2 * lineThickness);
    }
    //currently hardcoded, maybe add an output directory in config?
    String path = "/Users/coopermaira/Desktop/pyramidImage" + std::to_string(componentIndex) + "_" +
                  std::to_string(imagePyramid->scale) +
                  ".png";
    //only write pixels with information
    //imwrite(path, pyramidImage(Rect(left_offset, top_offset, width - left_offset - right_offset, height - top_offset - bottom_offset)));
    //debug_draw_voronoi(pyramidImage,subdiv,true);
    if (_fileName != "") {
      imwrite(_fileName, pyramidImage);
    } else {
      imwrite(path, pyramidImage);
    }
  }


  bool CompositeVoronoi::rebuildTile(Point2i tile, int sum) {
    std::string tileStr = std::to_string(tile.x) + "_" + std::to_string(tile.y);
    if (tileToSumNonZero[tileStr] < sum) {
      tileToSumNonZero[tileStr] = sum;
      return true;
    }
    return false;
  }

  void CompositeVoronoi::add_images_with_composite(std::vector<RegInfo *> new_info) {
    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<unsigned long> indexes;
    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i]->index);
    }
    std::vector<Image *> images = parent->get_image_ref(indexes);

    std::vector<Point2i> effectedTiles;
    for (int i = 0; i < images.size(); i++) {
      //calculate where the new image will be copied to in the composite
      Rect copyzone = Rect(new_info[i]->absoluteCoords.x - root_offset.x,
                           new_info[i]->absoluteCoords.y - root_offset.y, images[i]->width, images[i]->height);


      //add point to delaunay triangulation
      std::vector<Point2i> face;
      auto fShift = Point2f(new_info[i]->absoluteCoords.x, new_info[i]->absoluteCoords.y);

      if (add_point_to_delaunay_triangulation(fShift, images[i], face, false) == -1) {
        freshMask.copyTo(polyMaskOutput);
        continue;
      }

      //proceed with addition to composite
      images[i]->load_raw_from_disk();
      Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat, threeChannelPreallocated, COLOR_BayerBG2BGR);

      if (images[i]->label == Image::_2X) {
        cv::divide(threeChannelPreallocated, flat_field, threeChannelPreallocated, 1.0, CV_8U);
      }

      //add alpha channel now so cvMat can be turned into juce image via memcpy
      channels[0] = threeChannelPreallocated; //3 channel
      channels[1] = polyMaskOutput; //1 channel
      merge(channels, fourChannelPreallocated);

      fourChannelPreallocated.copyTo(composite(copyzone), polyMaskOutput);
      freshMask.copyTo(polyMaskOutput);
      images[i]->free_memory_RAW();

      //calculate effected tiles
      calculate_effected_tiles(face, effectedTiles, new_info[i]->absoluteCoords);
    }
    std::sort(effectedTiles.begin(), effectedTiles.end(), PointCompare<Point2i>());
    effectedTiles.erase(std::unique(effectedTiles.begin(), effectedTiles.end(), PointEquality<Point2i>()),
                        effectedTiles.end());


    tiledImageBounds = cv::Rect_<float>((long) root_offset.x, (long) root_offset.y, composite.cols, composite.rows);
    imagePyramid->level[0]->insertMatAtBase(composite, tiledImageBounds, effectedTiles);
    imagePyramid->bounds = imagePyramid->level[0]->bounds;

    parent->update_observers();
  }


  Composite::Composite(StreamCam *parent) : update_mutex(new Poco::FastMutex()), parent(parent),
                                            root_offset(0.0, 0.0),
                                            max_offset(0.0, 0.0) {
    flat_field = parent->flat_field2X;

    //local_quality_score = score_image_2X(4852,6464,2190);
  }


  void CompositeVoronoi::update_Bbox_no_composite(std::vector<RegInfo *> new_info) {
    bool update_box = false;

    //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.
    Vec2 temp_offset = root_offset;

    // If any new frames extend beyond the current extent, expand cv image dimensions
    for (int i = 0; i < new_info.size(); i++) {
      if (new_info[i]->absoluteCoords.x < root_offset.x) {
        update_box = true;
        root_offset.x = new_info[i]->absoluteCoords.x;
      }
      if (new_info[i]->absoluteCoords.y < root_offset.y) {
        update_box = true;
        root_offset.y = new_info[i]->absoluteCoords.y;
      }

      if (new_info[i]->absoluteCoords.x + parent->image_width > max_offset.x) {
        update_box = true;
        max_offset.x = new_info[i]->absoluteCoords.x + parent->image_width;
      }
      if (new_info[i]->absoluteCoords.y + parent->image_height > max_offset.y) {
        update_box = true;
        max_offset.y = new_info[i]->absoluteCoords.y + parent->image_height;
      }
    }
    if (update_box) {
      auto topLevelBeforeAdding = imagePyramid->level.back();
      unsigned int tile_size = imagePyramid->tile_size;
      unsigned int topLogicSize = imagePyramid->level.back()->getLogicSize();
      bool addedLevel = false;
      bool canResize = std::log2(tile_size) - imagePyramid->level.size() >= 2;

      while (canResize && (topLogicSize < max_offset.x - root_offset.x || topLogicSize < max_offset.y -
                           root_offset.y)) {
        addedLevel = true;
        unsigned int logic_size = 2 * topLogicSize;
        int levelWithinPyramid = imagePyramid->level.size();
        assert(pow(2, levelWithinPyramid) == logic_size / tile_size);
        imagePyramid->level.push_back(
          std::make_shared<TiledImage>(imagePyramid, tile_size, logic_size, levelWithinPyramid));
        topLogicSize = logic_size;
        canResize = std::log2(tile_size) - imagePyramid->level.size() >= 2;
      }
      if (addedLevel) {
        auto tl = topLevelBeforeAdding->getIJ(Point2f(root_offset.x, root_offset.y));
        auto br = topLevelBeforeAdding->getIJ(Point2f(max_offset.x, max_offset.y));
        for (int x = tl.x; x <= br.x; x++) {
          for (int y = tl.y; y <= br.y; y++) {
            if (topLevelBeforeAdding->tiles(x, y) != nullptr) {
              auto tile = Point2i(x, y);
              auto myLevelRegion = cv::Rect_<float>(tile.x * tile_size, tile.y * tile_size, tile_size,
                                                    tile_size);
              topLevelBeforeAdding->tileUpwards(tile, myLevelRegion,
                                                *topLevelBeforeAdding->tiles(tile.x, tile.y), Rect(0,0,tile_size,tile_size));
            }
          }
        }
      }
    }
  }


  void Composite::update_Bbox(std::vector<RegInfo *> new_info) {
    bool update_box = false;

    //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.
    Vec2 temp_offset = root_offset;

    // If any new frames extend beyond the current extent, expand cv image dimensions
    for (int i = 0; i < new_info.size(); i++) {
      if (new_info[i]->absoluteCoords.x < root_offset.x) {
        update_box = true;
        root_offset.x = new_info[i]->absoluteCoords.x;
      }
      if (new_info[i]->absoluteCoords.y < root_offset.y) {
        update_box = true;
        root_offset.y = new_info[i]->absoluteCoords.y;
      }

      if (new_info[i]->absoluteCoords.x + 6464 > max_offset.x) {
        update_box = true;
        max_offset.x = new_info[i]->absoluteCoords.x + 6464;
      }
      if (new_info[i]->absoluteCoords.y + 4852 > max_offset.y) {
        update_box = true;
        max_offset.y = new_info[i]->absoluteCoords.y + 4852;
      }
    }

    if (update_box) {
      //if we are updating the bounding box, create a new combined image and copy old image into the correct location
      Mat4b new_combined(int(max_offset.y - root_offset.y), int(max_offset.x - root_offset.x), Vec4b(0, 0, 0, 0));
      //Mat new_combined_z_buffer = cv::Mat::zeros(cv::Size(new_combined.cols, new_combined.rows), CV_16U);

      if (composite.data) {
        Rect copyzone = Rect(temp_offset.x - root_offset.x, temp_offset.y - root_offset.y, composite.cols,
                             composite.rows);

        composite.copyTo(new_combined(copyzone));
        //composite_z_buffer.copyTo(new_combined_z_buffer(copyzone));
      }

      composite = new_combined;
      //composite_z_buffer = new_combined_z_buffer;
      unsigned int topLogicSize = imagePyramid->level.back()->getLogicSize();
      while (topLogicSize < composite.rows || topLogicSize < composite.cols) {
        unsigned int tile_size = imagePyramid->level[0]->getTileSize();
        unsigned int logic_size = 2 * topLogicSize;
        int levelWithinPyramid = imagePyramid->level.size();
        assert(pow(2, levelWithinPyramid) == logic_size / tile_size);
        imagePyramid->level.push_back(
          std::make_shared<TiledImage>(imagePyramid, tile_size, logic_size, levelWithinPyramid));
        topLogicSize = logic_size;

        if (composite.data) {
          Mat temp;
          resize(composite, temp,
                 Size(composite.cols / pow(2, levelWithinPyramid),
                      composite.rows / pow(2, levelWithinPyramid)));
          tiledImageBounds = cv::Rect_<float>(root_offset.x, root_offset.y, composite.cols, composite.rows);
          imagePyramid->level.back()->insertMat(temp, tiledImageBounds);
        }
      }
    }
  };

  void Composite::add_images(std::vector<RegInfo *> new_info) {
    std::vector<unsigned long int> indexes;

    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i]->index);
    }

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<Image *> images = parent->get_image_ref(indexes);

    //this may need to be placed inside the below for loop if frames ever vary in size. For now it is here so the mask only needs to be built once
    cv::Size image_size(images[0]->width, images[0]->height);
    /*
    Mat mask = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);

    circle(mask, cv::Point(image_size.width/2, image_size.height/2), 2190, cv::Scalar(255), -1);
    */

    for (int i = 0; i < images.size(); i++) {
      //calculate where the new image will be copied to in the composite
      Rect copyzone = Rect(new_info[i]->absoluteCoords.x - root_offset.x,
                           new_info[i]->absoluteCoords.y - root_offset.y, images[i]->width, images[i]->height);

      //calculate which pixels of the new image will be copied into the composite
      Mat use_locations = local_quality_score > composite_z_buffer(copyzone);


      if (countNonZero(use_locations) == 0) {
        continue; //not contributing, don't bother loading from disk
      }

      images[i]->load_raw_from_disk();
      Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);
      cv::divide(image_Mat, flat_field, image_Mat, 1.0, CV_8U);

      //imwrite(images[i]->get_ImageFile().getBaseName()+".png", image_Mat);
      /*
      Mat temp;
      composite_z_buffer.copyTo(temp, copyzone);
          */

      /*
      Mat temp = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);
      image_Mat.copyTo(temp,use_locations);
      imwrite(images[i]->get_ImageFile().getBaseName() + ".png", temp);
      */

      image_Mat.copyTo(composite(copyzone), use_locations);
      local_quality_score.copyTo(composite_z_buffer(copyzone), use_locations);

      images[i]->free_memory_RAW();
    }
    /*
    imshow("display",composite);
    waitKey(10);
    */
  }

  void Composite::update(std::vector<RegInfo *> new_info) {
    update_Bbox(new_info);
    add_images(new_info);
  }

  Mat Composite::get_composite() {
    return composite;
  }

  void CompositeVoronoi::reset_image_as_polygon() {
    imageBoundsAsPolygon[0] = Point2i(0, 0);
    imageBoundsAsPolygon[1] = Point2i(image_size.width, 0);
    imageBoundsAsPolygon[2] = Point2i(image_size.width, image_size.height);
    imageBoundsAsPolygon[3] = Point2i(0, image_size.height);
  }

  long CompositeVoronoi::segment_yval_at_point(float xloc, Point2f p1, Point2f p2) {
    if (p1.x == p2.x) {
      return max(p1.y, p2.y);
    }
    return long((p1.y - p2.y) / (p1.x - p2.x) * (xloc - p1.x) + p1.y);
  }

  void CompositeVoronoi::remove_duplicates_without_sort(std::vector<Point2i> &vec) {
    auto new_last = vec.end() - 1;

    for (auto current = vec.begin(); current != new_last; ++current) {
      for (auto consider = current + 1; consider != new_last && consider != vec.end();) {
        if (consider->x == current->x && consider->y == current->y) {
          std::iter_swap(consider, new_last);
          new_last--;
        } else {
          consider++;
        }
      }
    }
    vec.erase(new_last + 1, vec.end());
  }

  cv::Mat Composite::score_image_2X(int rows, int cols, int radius) {
    cv::Mat img = cv::Mat::zeros(cv::Size(cols, rows), CV_16U);
    //cv::Mat img = cv::Mat_<uint16_t>(rows,cols);
    double idist, jdist, rad_sq;
    rad_sq = pow(radius, 2);
    double k = 0;
    for (int i = 0; i < radius; i++) {
      for (int j = 0; j < radius; j++) {
        int16_t x = rows / 2 - radius + i + 1;
        int16_t y = cols / 2 - radius + j + 1;
        /*
        //pyramid
        if (j < i) {
            img.at<uint16_t>(x,y) = j;
            img.at<uint16_t>(img.rows - x, y) = j;
            img.at<uint16_t>(x,img.cols - y) = j;
            img.at<uint16_t>(img.rows -x, img.cols - y) = j;
        }
        else {
            img.at<uint16_t>(x, y) = i;
            img.at<uint16_t>(img.rows - x, y) = i;
            img.at<uint16_t>(x, img.cols - y) = i;
            img.at<uint16_t>(img.rows - x, img.cols - y) = i;
        }
        */

        //cone
        img.at<uint16_t>(x, y) = radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
        img.at<uint16_t>(img.rows - x, y) =
            radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
        img.at<uint16_t>(x, img.cols - y) =
            radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
        img.at<uint16_t>(img.rows - x, img.cols - y) =
            radius - max(sqrt(pow(img.rows / 2 - x, 2) + pow(img.cols / 2 - y, 2)), 0.0);
      }
    }
    Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_16U);

    circle(mask, cv::Point(img.cols / 2, img.rows / 2), parent->scope_radius, cv::Scalar(1), -1);
    cv::Mat temp = mask.mul(img);
    imwrite("mask.png", temp);
    //imwrite("img.png", img);
    return temp;
  }


  void ImageToTileCopyRunnable::run() {
    std::cout << "deprecated method ImageToTileCopyRunnable::run()" << std::endl;
    assert(false);
    auto composite = parent->composites[component_membership];
    try {
      auto mask = composite->polyMaskOutput;
      auto imageMat = composite->fourChannelPreallocated;
      auto tileSize = composite->imagePyramid->level[0]->getTileSize();
      auto tileBox = cv::Rect_<float>(tileSize * tile.x, tileSize * tile.y, tileSize, tileSize);
      auto imageBox = cv::Rect_<float>(image->absoluteCoords.x, image->absoluteCoords.y, image->width,
                                       image->height);

      //composite->imagePyramid->insertTilesAtBase(imageMat, mask, imageBox, {tile});
    } catch (cv::Exception &e) {
      int k = 0;
    }

    composite->notify_job_complete();
  }

  void
  CompositeVoronoi::exclude_for_blur() {
    Subdiv2D dt;
    std::map<int, unsigned long> dtMembers;

    std::vector<unsigned long> image_indexes(delaunayMembers.size());
    int i = 0;
    for (auto item: delaunayMembers) {
      image_indexes[i] = item.second;
      i++;
    }

    auto images = parent->get_image_ref(image_indexes);
    blurVals.clear();
    blurVals.resize(images.size());
    for (int i = 0; i < images.size(); i++) {
      blurVals[i] = images[i]->motionBlur;
    }

    if (blurVals.size() > 0) {
      double sum = std::accumulate(std::begin(blurVals), std::end(blurVals), 0.0);
      double m = sum / blurVals.size();

      double accum = 0.0;
      std::for_each(std::begin(blurVals), std::end(blurVals), [&](const double d) {
        accum += (d - m) * (d - m);
      });

      double stdev = sqrt(accum / (blurVals.size() - 1));
      std::vector<Image *> res, rej;
      for (auto img: images) {
        if (img->motionBlur > m - 1.0 * stdev) {
          res.push_back(img);
        }
        //        else{
        //          rej.push_back(img->index);
        //        }
      }
      dt.initDelaunay(subdiv_Bbox.as_cvRect());
      for (auto i: res) {
        dtMembers[dt.insert(Point2f(i->absoluteCoords.x, i->absoluteCoords.y))] = i->index;
      }
      subdiv = dt;
      delaunayMembers = dtMembers;
    }
  }

  void CompositeVoronoi::build_system_from_DT(std::map<long, long> &systemIndexToFrameIndex,
                                              std::map<long, long> &frameIndexToSystemIndex, cv::Mat &A, cv::Mat &bx,
                                              cv::Mat &by, cv::Mat &x, cv::Mat &y) {
    std::vector<Vec4f> edges;
    std::vector<Vec2i> verticePairs;
    std::vector<Point2f> coords;

    subdiv.getEdgeList(edges);

    parent->resize_mmatch_mutex->readLock();
    int count1 = 0;
    int count2 = 0;
    for (int i = 0; i < edges.size(); i++) {
      //set point to shorten if statement
      auto ep = edges[i];
      auto x1 = subdiv_Bbox.max_x;
      auto x2 = subdiv_Bbox.min_x;
      auto y1 = subdiv_Bbox.max_y;
      auto y2 = subdiv_Bbox.min_y;


      //make sure edge ends are within bounding box
      if (ep[0] < x1 && ep[0] > x2 && ep[1] < y1 && ep[1] > y2 && ep[2] < x1 && ep[2] > x2 && ep[3] < y1 &&
          ep[3] > y2) {
        //find vertex IDs
        int vertId1 = subdiv.findNearest({ep[0], ep[1]});
        int vertId2 = subdiv.findNearest({ep[2], ep[3]});

        //verify vertices correspond to images added to composite
        auto val1 = delaunayMembers.count(vertId1);
        auto val2 = delaunayMembers.count(vertId2);

        if (val1 > 0 && val2 > 0) {
          count2++;
          auto image_idx1 = delaunayMembers[vertId1];
          auto image_idx2 = delaunayMembers[vertId2];
          auto m = parent->matchM.match[image_idx1][image_idx2];

          if (m == nullptr) {
            m = parent->matchM.match[image_idx2][image_idx1];
            image_idx2 = delaunayMembers[vertId1];
            image_idx1 = delaunayMembers[vertId2];
          }

          auto imageReg1 = parent->get_reg_ref(image_idx1);
          auto imageReg2 = parent->get_reg_ref(image_idx2);

          if (m != nullptr) {
            auto val = imageReg1->absoluteCoords.x - imageReg2->absoluteCoords.x - m->t_x;
            auto i1 = imageReg1->absoluteCoords.x;
            auto i2 = imageReg2->absoluteCoords.x;
            auto i3 = m->t_x;

            if (abs(val) < 200) {
              count1++;
              matchedEdges.push_back({image_idx1, image_idx2});

              if (image_idx1 != 0) {
                if (frameIndexToSystemIndex.find(image_idx1) == frameIndexToSystemIndex.end()) {
                  long val = frameIndexToSystemIndex.size();
                  frameIndexToSystemIndex[image_idx1] = val;
                  systemIndexToFrameIndex[val] = image_idx1;
                }
              }
              if (image_idx2 != 0) {
                long val = frameIndexToSystemIndex.size();
                if (frameIndexToSystemIndex.find(image_idx2) == frameIndexToSystemIndex.end()) {
                  frameIndexToSystemIndex[image_idx2] = val;
                  systemIndexToFrameIndex[val] = image_idx2;
                }
              }
            } else {
              int k = 0;
            }
          }
        }
      }
    }
    A = Mat::zeros(matchedEdges.size(), frameIndexToSystemIndex.size(), CV_64FC1);

    bx = Mat::zeros(matchedEdges.size(), 1, CV_64FC1);
    by = bx.clone();

    x = Mat::zeros(systemIndexToFrameIndex.size(), 1, CV_64FC1);
    y = x.clone();


    for (int i = 0; i < matchedEdges.size(); i++) {
      //build system (A in Ax - b)
      if (matchedEdges[i].first != memberImages[0].first->index) {
        A.at<double>(i, frameIndexToSystemIndex[matchedEdges[i].first]) = 1;
      }
      if (matchedEdges[i].second != memberImages[0].first->index) {
        A.at<double>(i, frameIndexToSystemIndex[matchedEdges[i].second]) = -1;
      }
      //build pairwise reg vector (b in Ax - b)
      auto pwr = parent->matchM.match[matchedEdges[i].first][matchedEdges[i].second];
      bx.at<double>(i) = pwr->t_x;
      by.at<double>(i) = pwr->t_y;
    }

    for (int i = 0; i < systemIndexToFrameIndex.size(); i++) {
      auto coords = parent->reg_results[systemIndexToFrameIndex[i]]->absoluteCoords;
      x.at<double>(i) = coords.x;
      y.at<double>(i) = coords.y;
    }
    int k = 0;
  }


  void CompositeVoronoi::perform_global_alignment(unsigned int flag, double closenessFactor) {
    //exclude_for_blur();

    //flag == 0 will pull delaunay edges. Flag == 1 will pull all possible overlaps < closenessFactor

    if (!needsAlignment) {
      return;
    }


    if (flag == 1) {
      std::vector<std::tuple<unsigned long, unsigned long, int, bool> > indexIndexEdgenumJobneeded;
      int edgeNumber = 0;
      for (int i = 0; i < memberImages.size() - 1; i++) {
        for (int j = i + 1; j < memberImages.size(); j++) {
          auto image1 = memberImages[j].first;
          auto image2 = memberImages[i].first;

          if (abs(image2->absoluteCoords.x - image1->absoluteCoords.x) < closenessFactor * image_size.width &&
              abs(image2->absoluteCoords.y - image1->absoluteCoords.y) < closenessFactor * image_size.
              height) {
            indexIndexEdgenumJobneeded.push_back({image1->index, image2->index, edgeNumber, true});
            //}
          }
        }
      }


      matchedEdges.resize(indexIndexEdgenumJobneeded.size(), {-1, -1});

      for (int i = 0; i < indexIndexEdgenumJobneeded.size(); i++) {
        auto edge = indexIndexEdgenumJobneeded[i];
        auto idx1 = std::get<0>(edge);
        auto idx2 = std::get<1>(edge);
        if ((idx1 == 367 && idx2 == 412) || (idx1 == 412 && idx2 == 367)) {
          int k = 0;
        }
        if (std::get<3>(edge)) {
          auto sm = new SingleMatchRunnable(parent, idx1, idx2, componentIndex, i, 0);
          matchableCount++;
          parent->JobQ->add_runnable(sm);
        } else {
          matchedEdges[i].first = idx1;
          matchedEdges[i].second = idx2;
        }
      }

      //from this point on is experimental use of all DT edges as constraints, updating constraints when
      //point is removed

      while (matchableCount > 0) {
        Poco::Thread::sleep(100);
      }
      matchedEdges.clear();

      std::vector<RegInfo *> new_info(memberImages.size());
      for (int i = 0; i < memberImages.size(); i++) {
        new_info[i] = memberImages[i].first->regInfo;
      }

      std::map<long, long> frameIndexToSystemIndex;
      std::map<long, long> systemIndexToFrameIndex;

      Mat A;
      Mat xpr;
      Mat ypr;
      Mat xac;
      Mat yac;

      self_reset();
      rebuild_DT_elementwise(new_info, true, false);
      build_system_from_DT(systemIndexToFrameIndex, frameIndexToSystemIndex, A, xpr, ypr, xac, yac);

      auto testValBefore = norm(A * xac - xpr);
      //solve problem for x and y
      std::map<long, long> temp;

      coopers_conjugate_gradient2(A, xpr, xac, 10000, 0.0000001, true, systemIndexToFrameIndex, 0.005, ypr, 0);
      coopers_conjugate_gradient2(A, ypr, yac, 10000, 0.0000001, true, systemIndexToFrameIndex, 0.005, xpr, 1);

      auto testValAfter = norm(A * xac - xpr);

      self_reset();
      parent->update_observers();
      parent->reg_results_mutex->readLock();
      std::vector<RegInfo *> newinfo;
      newinfo.push_back(parent->get_reg_ref(memberImages[0].first->index));
      double xDiff, yDiff;
      double xDiffMax = 0;
      double yDiffMax = 0;
      for (auto [i, elm]: systemIndexToFrameIndex) {
        auto ri = parent->get_reg_ref(systemIndexToFrameIndex[i]);

        xDiff = abs(ri->absoluteCoords.x - xac.at<double>(i));
        if (xDiff > xDiffMax) {
          xDiffMax = xDiff;
        }

        yDiff = abs(ri->absoluteCoords.y - yac.at<double>(i));
        if (yDiff > yDiffMax) {
          yDiffMax = yDiff;
        }

        //ri->absoluteCoords.x = xac.at<double>(i);
        //ri->absoluteCoords.y = yac.at<double>(i);

        newinfo.push_back(ri);
      }

      parent->reg_results_mutex->unlock();

      update(newinfo);

      int k = 0;

      return;
    } else if (flag == 0) {
      //collect list of all edges.
      std::vector<Vec4f> edges;
      std::vector<Vec2i> verticePairs;
      std::vector<Point2f> coords;

      subdiv.getEdgeList(edges);
      matchedEdges.resize(edges.size(), {-1, -1}); //preallocated to avoid mutex

      //this mutex is locked here because each job created in the following loop needs this mutex to be read locked.
      //They will not cause a reallocation because these images have already had initial matches, meaning the match
      //matrix has already been resized to accommodate them. Its unlocked at end of loop

      //collect list of all vertices that share an edge
      int count = 0;
      for (int i = 0; i < edges.size(); i++) {
        //set point to shorten if statement
        auto ep = edges[i];
        auto x1 = subdiv_Bbox.max_x;
        auto x2 = subdiv_Bbox.min_x;
        auto y1 = subdiv_Bbox.max_y;
        auto y2 = subdiv_Bbox.min_y;


        //make sure edge ends are within bounding box
        if (ep[0] < x1 && ep[0] > x2 && ep[1] < y1 && ep[1] > y2 && ep[2] < x1 && ep[2] > x2 && ep[3] < y1 &&
            ep[3] > y2) {
          //find vertex IDs
          int vertId1 = subdiv.findNearest({ep[0], ep[1]});
          int vertId2 = subdiv.findNearest({ep[2], ep[3]});

          //verify vertices correspond to images added to composite
          auto val1 = delaunayMembers.count(vertId1);
          auto val2 = delaunayMembers.count(vertId2);

          if (val1 > 0 && val2 > 0) {
            auto image_idx1 = delaunayMembers[vertId1];
            auto image_idx2 = delaunayMembers[vertId2];

            //create matchable job
            matchableCount++;
            auto sm = new SingleMatchRunnable(parent, delaunayMembers[vertId1], delaunayMembers[vertId2],
                                              componentIndex, i, 0);
            parent->JobQ->add_runnable(sm);
          }
        }
      }
    } //end edge type job assignment

    //wait until these jobs have completed
    while (matchableCount > 0) {
      Poco::Thread::sleep(100);
    }


    //give each frame index a linear system index.
    std::map<long, long> frameIndexToSystemIndex;
    std::map<long, long> systemIndexToFrameIndex;
    for (int i = 0; i < matchedEdges.size();) {
      if (matchedEdges[i].first == -1) {
        //just because a delaunay edge exists between two frames doesn't mean they actually overlap. Some edges may
        //be far enough away that a registration between them is impossible, but a delaunay edge still exists. Delete
        //these so that an accurate count of system equations can be made.
        matchedEdges.erase(matchedEdges.begin() + i);
        continue;
      }
      if (matchedEdges[i].first != memberImages[0].first->index) {
        if (frameIndexToSystemIndex.find(matchedEdges[i].first) == frameIndexToSystemIndex.end()) {
          long val = frameIndexToSystemIndex.size();
          frameIndexToSystemIndex[matchedEdges[i].first] = val;
          systemIndexToFrameIndex[val] = matchedEdges[i].first;
        }
      }
      if (matchedEdges[i].second != memberImages[0].first->index) {
        long val = frameIndexToSystemIndex.size();
        if (frameIndexToSystemIndex.find(matchedEdges[i].second) == frameIndexToSystemIndex.end()) {
          frameIndexToSystemIndex[matchedEdges[i].second] = val;
          systemIndexToFrameIndex[val] = matchedEdges[i].second;
        }
      }
      i++;
    }
    //no constraints means no linear system to minimize
    if (matchedEdges.size() == 0) {
      return;
    }

    //minimize norm2(Ax-b).
    Mat A = Mat::zeros(matchedEdges.size(), frameIndexToSystemIndex.size(), CV_64FC1);
    Mat xpr = Mat::zeros(matchedEdges.size(), 1, CV_64FC1);
    Mat ypr = xpr.clone();
    Mat xprLP = xpr.clone();
    Mat yprLP = ypr.clone();

    double valtest1 = 0;
    unsigned long ind1, ind2;
    std::vector<int> bins1;
    for (int i = 0; i < matchedEdges.size(); i++) {
      //build system (A in Ax - b)
      if (matchedEdges[i].first != 0) {
        A.at<double>(i, frameIndexToSystemIndex[matchedEdges[i].first]) = 1;
      } else {
        xprLP.at<double>(i, 0) += root_offset.x;
        yprLP.at<double>(i, 0) += root_offset.y;
      }
      if (matchedEdges[i].second != 0) {
        A.at<double>(i, frameIndexToSystemIndex[matchedEdges[i].second]) = -1;
      } else {
        xprLP.at<double>(i, 0) += -1 * root_offset.x;
        yprLP.at<double>(i, 0) += -1 * root_offset.y;
      }
      //build pairwise reg vector (b in Ax - b)
      auto pwr = parent->matchM.match[matchedEdges[i].first][matchedEdges[i].second];
      xpr.at<double>(i, 0) = pwr->t_x;
      ypr.at<double>(i, 0) = pwr->t_y;
      xprLP.at<double>(i, 0) += pwr->t_x;
      yprLP.at<double>(i, 0) += pwr->t_y;

      auto verify1 = parent->reg_results[matchedEdges[i].first]->absoluteCoords;
      auto verify2 = parent->reg_results[matchedEdges[i].second]->absoluteCoords;
      auto val = abs(verify1.x - verify2.x - pwr->t_x);

      if (val > valtest1) {
        valtest1 = val;
        ind1 = matchedEdges[i].first;
        ind2 = matchedEdges[i].second;
      }
      if (val > 35) {
        int k = 0;
      }
      if ((int) val >= bins1.size()) {
        bins1.resize((int) val + 1, 0);
        bins1[(int) val] = 1;
      } else {
        bins1[(int) val]++;
      }
    }

    //build absolute coord/solution vector (x in Ax - b)
    Mat xac = Mat::zeros(systemIndexToFrameIndex.size(), 1, CV_64FC1);
    Mat yac = xac.clone();

    for (int i = 0; i < systemIndexToFrameIndex.size(); i++) {
      auto coords = parent->reg_results[systemIndexToFrameIndex[i]]->absoluteCoords;
      xac.at<double>(i) = coords.x;
      yac.at<double>(i) = coords.y;
    }

    /*some comments:
     * Because of the geometry of the delaunay triangulation, A^T * A is guaranteed to be symmetric positive definite.
     * For this reason, minimizing the 2norm of Ax-b is well suited for conjugate gradient. I have written my own
     * easy implementation and compared it in time trials to calculating the moore-pemrose inverse. for 100 iterations,
     * CG takes 3 miliseconds while the MP inverse takes 11. Results are comparable.
     *
    start = std::chrono::high_resolution_clock::now();
    Mat At = A.t();
    Mat MPI = (At*A).inv()*At*xpr;
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
     */

    auto testValBefore = norm(A * xac - xpr);
    //solve problem for x and y
    std::map<long, long> temp;

    coopers_conjugate_gradient(A, xpr, xac, 10000, 0.0000001, false, systemIndexToFrameIndex, 0.005, ypr);
    coopers_conjugate_gradient(A, ypr, yac, 10000, 0.0000001, false, systemIndexToFrameIndex, 0.005, xpr);

    auto testValAfter = norm(A * xac - xpr);
    int k = 0;
    /*

 //compute LP problem to minimize inf norm instead of 2 norm
    Mat c = Mat::zeros(A.cols + 1, 1, CV_64FC1);
    c.at<double>(A.cols) = -1.0;

    Mat G = Mat::zeros(2 * A.rows, A.cols + 2, CV_64FC1);
    A.copyTo(G(Rect(0, 0, A.cols, A.rows)));
    A.copyTo(G(Rect(0, A.rows, A.cols, A.rows)));
    G(Rect(0, A.rows, A.cols, A.rows)) *= -1;
    for (int i = 0; i < G.rows; i++) {
      G.at<double>(i, A.cols) = -1;
    }

    xprLP.copyTo(G(Rect(A.cols + 1, 0, 1, xpr.rows)));
    xprLP.copyTo(G(Rect(A.cols + 1, xpr.rows, 1, xpr.rows)));
    G(Rect(A.cols + 1, xpr.rows, 1, xprLP.rows)) *= -1;

    Mat xLP;
    auto result = cv::solveLP(c, G, xLP);


    Mat hy = Mat::zeros(2*A.rows,1,CV_64FC1);
    ypr.copyTo(hy(Rect(0,0,1,ypr.rows)));
    ypr.copyTo(hy(Rect(0,ypr.rows,1,ypr.rows)));
    hy(Rect(0,ypr.rows,1,ypr.rows)) *= -1;

*/
    auto start = std::chrono::high_resolution_clock::now();


    parent->update_observers();
    parent->reg_results_mutex->readLock();
    std::vector<RegInfo *> newinfo;

    newinfo.push_back(memberImages[0].first->regInfo);
    for (auto [i, elm]: systemIndexToFrameIndex) {
      parent->reg_results[systemIndexToFrameIndex[i]]->absoluteCoords.x = xac.at<double>(i);
      parent->reg_results[systemIndexToFrameIndex[i]]->absoluteCoords.y = yac.at<double>(i);
      newinfo.push_back(parent->reg_results[systemIndexToFrameIndex[i]]);
    }
    parent->reg_results_mutex->unlock();

    self_reset();
    update(newinfo, true);
    //rebuild_DT_elementwise(newinfo, true,false);
    //create_and_submit_rebuild_jobs();


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    needsAlignment = false;
  }

  void CompositeVoronoi::coopers_conjugate_gradient2(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon,
                                                     bool shouldCleanData,
                                                     std::map<long, long> &systemIndexToFrameIndex,
                                                     double epsilonClean, cv::Mat bOther, int flag) {
    int maxNumberRemoved = systemIndexToFrameIndex.size() / 4;

    std::vector<long> indexesRemoved;
    cv::Mat ATranspose = A.t();
    cv::Mat ATA = ATranspose * A;

    cv::Mat r = ATranspose * b - (ATA * x);
    cv::Mat p = r.clone();
    double n0 = cv::norm(A * x - b);
    double n1;

    int removeCount = 0;
    for (int i = 0; i < steps; i++) {
      double rdr = r.dot(r);
      Mat ATAp = ATA * p;
      double stepSize = rdr / (p.dot(ATAp));

      x += stepSize * p;

      n1 = cv::norm(A * x - b);

      r -= stepSize * ATAp;
      double adjustment = r.dot(r) / rdr;
      p = r + adjustment * p;

      std::cout << "step " + std::to_string(i) + "   norm difference " + std::to_string(abs(n0 - n1)) +
          "   norm of residual " + std::to_string(abs(n1)) << std::endl;

      if (removeCount < maxNumberRemoved && shouldCleanData && abs(n0 - n1) < cv::norm(n1) * epsilonClean) {
        removeCount++;

        clean_data(A, b, bOther, x, systemIndexToFrameIndex);

        std::vector<RegInfo *> new_info(systemIndexToFrameIndex.size() + 1);
        new_info[0] = parent->get_reg_ref(memberImages[0].first->index);
        double diffmax = 0;
        int ii = 1;
        for (auto el: systemIndexToFrameIndex) {
          auto ni = parent->get_reg_ref(el.second);
          auto val = x.at<double>(el.first);
          new_info[ii] = ni;
          double diff;
          if (flag == 0) {
            //change x
            diff = abs(new_info[ii]->absoluteCoords.x - val);
            new_info[ii]->absoluteCoords.x = val;
          } else {
            //change y
            diff = abs(new_info[ii]->absoluteCoords.y - val);
            new_info[ii]->absoluteCoords.y = val;
          }
          if (diff > diffmax) {
            diffmax = diff;
          }
          ii++;
        }

        //reset everything and rebuild
        systemIndexToFrameIndex.clear();
        auto f2s = systemIndexToFrameIndex;

        A.release();
        b.release();
        x.release();
        bOther.release();
        Mat xOther;

        self_reset();

        rebuild_DT_elementwise(new_info, true, false);
        build_system_from_DT(systemIndexToFrameIndex, f2s, A, b, bOther, x, xOther);

        ATA = A.t() * A;
        r = A.t() * b - (ATA * x);
        p = r.clone();
      }

      if (abs(n0 - n1) < epsilon || n1 < epsilon) {
        std::vector<RegInfo *> new_info(systemIndexToFrameIndex.size() + 1);
        new_info[0] = parent->get_reg_ref(memberImages[0].first->index);
        double diffmax = 0;
        int ii = 1;
        for (auto el: systemIndexToFrameIndex) {
          auto ni = parent->get_reg_ref(el.second);
          auto val = x.at<double>(el.first);
          new_info[ii] = ni;
          double diff;
          if (flag == 0) {
            //change x
            diff = abs(new_info[ii]->absoluteCoords.x - val);
            new_info[ii]->absoluteCoords.x = val;
          } else {
            //change y
            diff = abs(new_info[ii]->absoluteCoords.y - val);
            new_info[ii]->absoluteCoords.y = val;
          }
          if (diff > diffmax) {
            diffmax = diff;
          }
          ii++;
        }
        break;
      }

      n0 = n1;
    }
  }


  void CompositeVoronoi::coopers_conjugate_gradient(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon,
                                                    bool shouldCleanData,
                                                    std::map<long, long> &systemIndexToFrameIndex,
                                                    double epsilonClean,
                                                    cv::Mat bOther) {
    int maxNumberRemoved = systemIndexToFrameIndex.size() / 4;

    std::vector<long> indexesRemoved;
    cv::Mat ATranspose = A.t();
    cv::Mat ATA = ATranspose * A;
    cv::Mat ATb = ATranspose * b;

    cv::Mat r = ATb - (ATA * x);
    cv::Mat p = r.clone();
    double n0 = cv::norm(A * x - b);
    double n1;

    int removeCount = 0;
    for (int i = 0; i < steps; i++) {
      double rdr = r.dot(r);
      Mat ATAp = ATA * p;
      double stepSize = rdr / (p.dot(ATAp));

      x += stepSize * p;

      n1 = cv::norm(A * x - b);

      r -= stepSize * ATAp;
      double adjustment = r.dot(r) / rdr;
      p = r + adjustment * p;

      std::cout << "step " + std::to_string(i) + "   norm difference " + std::to_string(abs(n0 - n1)) +
          "   norm of residual " + std::to_string(abs(n1)) << std::endl;

      if (removeCount < maxNumberRemoved && shouldCleanData && abs(n0 - n1) < cv::norm(n1) * epsilonClean) {
        removeCount++;
        clean_data(A, b, bOther, x, systemIndexToFrameIndex);
        ATA = A.t() * A;
        r = A.t() * b - (ATA * x);
        p = r.clone();
      }

      if (abs(n0 - n1) < epsilon || n1 < epsilon) {
        break;
      }

      n0 = n1;
    }
  }


  void CompositeVoronoi::clean_data(cv::Mat A, cv::Mat b, cv::Mat bOther, cv::Mat x,
                                    std::map<long, long> &systemIndexToFrameIndex) {
    Mat r = A * x - b;
    Mat e = r.mul(r);
    Mat bins = Mat::zeros(x.rows, 1, CV_64FC1);
    for (int i = 0; i < A.cols; i++) {
      for (int j = 0; j < A.rows; j++) {
        if (A.at<double>(j, i) != 0) {
          bins.at<double>(i) += e.at<double>(j);
        }
      }
    }
    //
    //    for(int i = 0; i < bins.rows; i++){
    //      std::cout<<std::to_string(bins.at<double>(i))<<std::endl;
    //    }

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc(bins, &minVal, &maxVal, &minLoc, &maxLoc);
    std::cout << std::to_string(maxVal) + " " + std::to_string(pow(cv::norm(r), 2)) << std::endl;

    std::vector<double> bins2(bins.rows);
    for (int i = 0; i < bins.rows; i++) {
      bins2[i] = bins.at<double>(i);
    }

    double sum = std::accumulate(std::begin(bins2), std::end(bins2), 0.0);
    double m = sum / bins2.size();

    double accum = 0.0;
    std::for_each(std::begin(bins2), std::end(bins2), [&](const double d) {
      accum += (d - m) * (d - m);
    });

    double stdev = sqrt(accum / (bins2.size() - 1));
    double distFromMean = (maxVal - m) / stdev;
    std::cout << "distance from mean " + std::to_string(distFromMean) << std::endl;
    //if(distFromMean > 4){
    removeCount++;
    long indexRemoved = maxLoc.y;

    //    for (int i = 0; i < A.rows; i++) {
    //      for (int j = 0; j < A.cols; j++) {
    //        std::cout << std::to_string(A.at<double>(i, j)) + " ";
    //      }
    //      std::cout << "        " + std::to_string(b.at<double>(i)) << std::endl;
    //    }

    for (int i = 0; i < A.rows; i++) {
      if (A.at<double>(i, indexRemoved) != 0) {
        b.at<double>(i) = 0;
        if (bOther.rows > 0) {
          bOther.at<double>(i) = 0;
        }
        for (int ii = 0; ii < A.cols; ii++) {
          A.at<double>(i, ii) = 0;
        }
        systemIndexToFrameIndex.erase(indexRemoved);
      }
    }
    //    std::cout << std::endl;
    //    for (int i = 0; i < A.rows; i++) {
    //      for (int j = 0; j < A.cols; j++) {
    //        std::cout << std::to_string(A.at<double>(i, j)) + " ";
    //      }
    //      std::cout << "       " + std::to_string(b.at<double>(i)) << std::endl;
    //    }


    int k = 0;

    //}
  }
}
