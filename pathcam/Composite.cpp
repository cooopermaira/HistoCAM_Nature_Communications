//
//  Composite.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/24/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam {

  CompositeVoronoi::CompositeVoronoi(StreamCam *parent, cv::Size image_size, unsigned int component_index) : Composite(
      parent), componentIndex(component_index), wakeEvent(true), image_size(image_size) {
    imagePyramid.reset(new MRTiledImage);
    std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(imagePyramid);
    imagePyramid->level.push_back(current);
    parent->MRimage->add(imagePyramid);

    subdiv_Bbox = Bbox(-50000, -50000, 50000, 50000);
    subdiv.initDelaunay(subdiv_Bbox.as_cvRect());

    circleMask = cv::Mat::zeros(image_size, CV_8U);
    cv::circle(circleMask, cv::Point(image_size.width / 2, image_size.height / 2), parent->scope_radius, cv::Scalar(1),
               -1);

    channels.resize(2);

    imageBoundsAsPolygon.resize(4);
    reset_image_as_polygon();

    polyMaskOutput = cv::Mat::zeros(image_size, CV_8U);
    freshMask = polyMaskOutput.clone();

    //testing conjugate gradient data cleaner
    Mat A = Mat::zeros(8, 4, CV_64FC1);

    Mat bx = Mat::zeros(8, 1, CV_64FC1);
    Mat by = bx.clone();

    Mat x = Mat::zeros(4, 1, CV_64FC1);
    Mat y = x.clone();

    A.at<double>(0, 0) = 1;
    A.at<double>(1, 0) = -1;
    A.at<double>(1, 1) = 1;
    A.at<double>(2, 1) = 1;
    A.at<double>(3, 2) = 1;
    A.at<double>(4, 1) = -1;
    A.at<double>(4, 2) = 1;
    A.at<double>(5, 3) = 1;
    A.at<double>(6, 3) = 1;
    A.at<double>(6, 1) = -1;
    A.at<double>(7, 3) = 1;
    A.at<double>(7, 0) = -1;

    bx.at<double>(0) = 2;
    bx.at<double>(1) = 2;
    bx.at<double>(2) = 4;
    bx.at<double>(3) = 5;
    bx.at<double>(4) = 1;
    bx.at<double>(5) = 4;
    bx.at<double>(6) = 1;
    bx.at<double>(7) = 1;

    by.at<double>(0) = -1;
    by.at<double>(1) = 1.01;
    by.at<double>(2) = 0;
    by.at<double>(3) = -2;
    by.at<double>(4) = -1.95;
    by.at<double>(5) = 4;
    by.at<double>(6) = 5;
    by.at<double>(7) = 4;

    x.at<double>(0) = 2;
    x.at<double>(1) = 4;
    x.at<double>(2) = 5;
    x.at<double>(3) = 3;

    y.at<double>(0) = -1.5;
    y.at<double>(1) = .5;
    y.at<double>(2) = -2;
    y.at<double>(3) = 4;


    std::map<long, long> systemMap;
    for (long i = 0; i < x.rows; i++) {
      systemMap[i] = i;
    }

    //coopers_conjugate_gradient(A, bx, x, 100, 0.00001, true, 0.05, systemMap,by);



    //coopers_conjugate_gradient(A, by, y, 100, 0.00001, true, 0.05, systemMap,bx);
    int k = 0;
  }


  void CompositeVoronoi::update(std::vector<RegInfo> new_info) {
    update_mutex->lock();

    if (new_info.size() > 1) {
      // this shuffle is very important for reducing image count in the DT.
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(new_info), std::end(new_info), rng);
    }

    update_Bbox_no_composite(new_info);
    expand_subdiv(new_info);
    add_images_no_composite(new_info);

    update_mutex->unlock();
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


  void CompositeVoronoi::expand_subdiv(std::vector<RegInfo> new_info) {
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
      std::vector<std::vector<Point2f>> facets;
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

  void CompositeVoronoi::rebuild_DT_elementwise(std::vector<RegInfo> new_info) {
    if (new_info.size() > 1) {
      // this shuffle is very important for reducing image count in the DT.
      auto rng = std::default_random_engine{};
      std::shuffle(std::begin(new_info), std::end(new_info), rng);
    }

    for (auto ni: new_info) {
      std::vector<Point2i> face;
      auto fShift = Point2f(ni.absoluteCoords.x, ni.absoluteCoords.y);
      auto img = parent->get_image_ref(ni.index);
      img->absoluteCoords = ni.absoluteCoords;
      auto res = add_point_to_delaunay_triangulation(fShift, img, face);
      freshMask.copyTo(polyMaskOutput);
    }
  }


  int CompositeVoronoi::add_point_to_delaunay_triangulation(cv::Point2f _point, pathCam::Image *_image,
                                                            std::vector<Point2i> &_face) {
    //make copy of subdiv incase we decide not to use new point
    Subdiv2D tempSubdiv(subdiv);

    //add new point
    int vertxId = subdiv.insert(_point);
    //auto image = parent->get_image_ref(_image_index);

    //get voronoi facets for only this face
    std::vector<std::vector<Point2f>> facets;
    std::vector<Point2f> centers;
    subdiv.getVoronoiFacetList({vertxId}, facets, centers);

    //shift and recast
    for (auto &ii: facets[0]) {//we have pulled only one face so facets has only 1 element
      ii.x -= centers[0].x;
      ii.x += image_size.width / 2;
      ii.y -= centers[0].y;
      ii.y += image_size.height / 2;
      _face.push_back((Point2i) ii);
    }

    //build polygon mask for new point
    cv::fillConvexPoly(polyMaskOutput, _face, cv::Scalar(255));

    //test for exclusion of frame via rollback
    int nonzeroMin;
    if (_image->label == Image::_2X) {
      polyMaskOutput = polyMaskOutput.mul(circleMask);
      nonzeroMin = parent->scope_radius * parent->scope_radius * 3.14 * 0.20;
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
    memberImages.push_back({_image, true});
    delaunayMembers.insert({vertxId, _image->index});
    return vertxId;
  }

  void CompositeVoronoi::create_and_submit_rebuild_jobs() {
    for (auto el : delaunayMembers){
      auto rr = new RebuildRunnable(this,el.first,el.second);
      parent->cm->rebuildJobsOutstanding++;
      parent->JobQ->add_runnable(rr);
    }
  }

  void CompositeVoronoi::add_images_no_composite(std::vector<RegInfo> new_info) {
    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<unsigned long> indexes;
    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i].index);
    }
    std::vector<Image *> images = parent->get_image_refs(indexes);

    for (int i = 0; i < images.size(); i++) {
      images[i]->component_membership = componentIndex;
      //add point to delaunay triangulation
      std::vector<Point2i> face;
      auto fShift = Point2f(new_info[i].absoluteCoords.x, new_info[i].absoluteCoords.y);
      auto res = add_point_to_delaunay_triangulation(fShift, images[i], face);

      //res is {vertexId,maskId}
      if (res == -1) { continue; }
      images[i]->vertexId = res;
      images[i]->absoluteCoords = new_info[i].absoluteCoords;

      //indicate that a new image has been added since last global alignment
      needsAlignment = true;

      //calculate effected tiles
      std::vector<Point2i> effectedTiles;
      calculate_effected_tiles(face, effectedTiles, new_info[i].absoluteCoords);

      //build image with alpha channel

      images[i]->load_raw_from_disk();
      Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat, threeChannelPreallocated, COLOR_BayerBG2BGR);
      images[i]->free_memory_RAW();

      if (images[i]->label == Image::_2X) {//flat field correction if needed
        auto center = Point2i(image_size.width / 2, image_size.height / 2);
        auto bb = Rect(center.x - parent->scope_radius - 10, center.y - parent->scope_radius - 10,
                       2 * parent->scope_radius + 20, 2 * parent->scope_radius + 20);
        cv::divide(threeChannelPreallocated(bb), flat_field(bb), threeChannelPreallocated(bb), 1.0, CV_8U);
      }

      channels[0] = threeChannelPreallocated; //3 channel
      channels[1] = polyMaskOutput;           //alpha channel
      merge(channels, fourChannelPreallocated);

      //debug_write_contribution_on_grid("test1.png",images[i]->absoluteCoords,fourChannelPreallocated,polyMaskOutput);
      int k = 0;
      for (auto tile: effectedTiles) {
        try {
          auto mask = polyMaskOutput;
          auto imageMat = fourChannelPreallocated;
          auto tileSize = imagePyramid->level[0]->getTileSize();
          auto tileBox = cv::Rect_<float>(tileSize * tile.x, tileSize * tile.y, tileSize, tileSize);
          auto imageBox = cv::Rect_<float>(images[i]->absoluteCoords.x, images[i]->absoluteCoords.y, images[i]->width,
                                           images[i]->height);

          imagePyramid->level[0]->inserTileAtBase(imageMat, mask, imageBox, {tile});
        }
        catch (cv::Exception &e) {
          int k = 0;
        }

      }


      //update pyramid bounds and observer, reset mask

      imagePyramid->bounds = imagePyramid->level[0]->bounds;
      if(imagePyramid->scale > 0){
        float x = (imagePyramid->offset.x + new_info.back().absoluteCoords.x) * imagePyramid->scale;
        float y = (imagePyramid->offset.y + new_info.back().absoluteCoords.y) * imagePyramid->scale;
        float w = parent->image_width * imagePyramid->scale;
        float h = parent->image_height * imagePyramid->scale;

        parent->update_last_frame(Rect_<float>(x,y,w,h),images.back()->label == Image::_2X);
      }
      parent->update_observers();
      freshMask.copyTo(polyMaskOutput);
    }
  }

  void CompositeVoronoi::add_images_multithread(std::vector<RegInfo> new_info) {

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<unsigned long> indexes;
    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i].index);
    }
    std::vector<Image *> images = parent->get_image_refs(indexes);

    for (int i = 0; i < images.size(); i++) {

      //add point to delaunay triangulation
      std::vector<Point2i> face;
      auto fShift = Point2f(new_info[i].absoluteCoords.x, new_info[i].absoluteCoords.y);
      auto res = add_point_to_delaunay_triangulation(fShift, images[i], face);

      //res is {vertexId,maskId}
      if (res == -1) { continue; }
      images[i]->vertexId = res;
      images[i]->absoluteCoords = new_info[i].absoluteCoords;

      //calculate effected tiles
      std::vector<Point2i> effectedTiles;
      calculate_effected_tiles(face, effectedTiles, new_info[i].absoluteCoords);

      //build image with alpha channel

      images[i]->load_raw_from_disk();
      Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
      cvtColor(image_Mat, threeChannelPreallocated, COLOR_BayerBG2BGR);

      images[i]->free_memory_RAW();

      if (images[i]->label == Image::_2X) {//flat field correction if needed
        cv::divide(threeChannelPreallocated, flat_field, threeChannelPreallocated, 1.0, CV_8U);
      }

      channels[0] = threeChannelPreallocated; //3 channel
      channels[1] = polyMaskOutput;           //alpha channel
      merge(channels, fourChannelPreallocated);

      //build and run copy runnable
      for (auto tile: effectedTiles) {
        jobCount++;
        auto cr = new ImageToTileCopyRunnable(parent, images[i], componentIndex, tile, 0);
        parent->JobQ->add_runnable(cr);
      }

      //wait till jobs have processed
      wakeEvent.wait();

      //update pyramid bounds and observer, reset mask
      imagePyramid->bounds = imagePyramid->level[0]->bounds;
      parent->update_observers();
      freshMask.copyTo(polyMaskOutput);
    }
  }

  void CompositeVoronoi::save_pyramid_as_image() {
    auto rootoffsetPoint = Point2f(root_offset.x, root_offset.y);
    auto maxOffsetPoint = Point2f(max_offset.x, max_offset.y);
    auto level = imagePyramid->level[0];
    auto ul = imagePyramid->level[0]->getIJ(rootoffsetPoint);
    auto lr = imagePyramid->level[0]->getIJ(maxOffsetPoint);
    int tile_size = level->getTileSize();
    int width = (lr.x + 1 - ul.x) * tile_size;
    width = std::abs(width);
    int height = (lr.y + 1 - ul.y) * tile_size;
    height = std::abs(height);
    Size pyramidSize = Size(width, height);
    Mat pyramidImage = Mat(pyramidSize, CV_8UC4);
    //we need to shift the x and y tiles so we aren't writing to negtive coordinates
    int x_offset = -ul.x;
    int y_offset = -ul.y;
    //for removing excess empty pixels around edge tiles
    int left_offset = std::abs(ul.x * tile_size - rootoffsetPoint.x);
    int top_offset = std::abs(ul.y * tile_size - rootoffsetPoint.y);
    int right_offset = std::abs(lr.x * tile_size - maxOffsetPoint.x);
    int bottom_offset = std::abs(lr.y * tile_size - maxOffsetPoint.y);
    for (int i = ul.x; i <= lr.x; i++) {
      for (int j = ul.y; j <= lr.y; j++) {
        Mat tile = level->getTile(i, j);
        //imwrite(std::to_string(componentIndex) + "_" + std::to_string(i) + "_" + std::to_string(j) + ".png", tile);
        tile.copyTo(pyramidImage(Rect((i + x_offset) * tile.cols, (j + y_offset) * tile.rows, tile.cols, tile.rows)));
      }
    }
    //currently hardcoded, maybe add an output directory in config?
    String path = "pyramidImage" + std::to_string(componentIndex) + "_"+std::to_string(imagePyramid->scale)+ ".png";
    //only write pixels with information
    //imwrite(path, pyramidImage(Rect(left_offset, top_offset, width - left_offset - right_offset, height - top_offset - bottom_offset)));
    imwrite(path,pyramidImage);
  }

  void CompositeVoronoi::add_images_with_composite(std::vector<RegInfo> new_info) {

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<unsigned long> indexes;
    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i].index);
    }
    std::vector<Image *> images = parent->get_image_refs(indexes);

    std::vector<Point2i> effectedTiles;
    for (int i = 0; i < images.size(); i++) {

      //calculate where the new image will be copied to in the composite
      Rect copyzone = Rect(new_info[i].absoluteCoords.x - root_offset.x,
                           new_info[i].absoluteCoords.y - root_offset.y, images[i]->width, images[i]->height);


      //add point to delaunay triangulation
      std::vector<Point2i> face;
      auto fShift = Point2f(new_info[i].absoluteCoords.x, new_info[i].absoluteCoords.y);

      if (add_point_to_delaunay_triangulation(fShift, images[i], face) == -1) {
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
      channels[1] = polyMaskOutput;              //1 channel
      merge(channels, fourChannelPreallocated);

      fourChannelPreallocated.copyTo(composite(copyzone), polyMaskOutput);
      freshMask.copyTo(polyMaskOutput);
      images[i]->free_memory_RAW();

      //calculate effected tiles
      calculate_effected_tiles(face, effectedTiles, new_info[i].absoluteCoords);
    }
    std::sort(effectedTiles.begin(), effectedTiles.end(), PointCompare<Point2i>());
    effectedTiles.erase(std::unique(effectedTiles.begin(), effectedTiles.end(), PointEquality<Point2i>()),
                        effectedTiles.end());


    tiledImageBounds = cv::Rect_<float>((long) root_offset.x, (long) root_offset.y, composite.cols, composite.rows);
    imagePyramid->level[0]->insertMatAtBase(composite, tiledImageBounds, effectedTiles);
    imagePyramid->bounds = imagePyramid->level[0]->bounds;

    parent->update_observers();
  }


  void CompositeVoronoi::calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<Point2i> &result,
                                                  Vec2 absCoord) {
    std::vector<Point2i> tileIndices;
    std::map<int, std::vector<float>> tilesByColumn;

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

        //if((enterColumn >= yPixelBoundLow || exitColumn >= yPixelBoundLow) && (enterColumn <= yPixelBoundHigh || exitColumn <= yPixelBoundHigh)) {
        //enterColumn = max(enterColumn, yPixelBoundLow);
        //enterColumn = min(enterColumn, yPixelBoundHigh);
        tilesByColumn[j].push_back((float) enterColumn);

        //exitColumn = max(exitColumn, yPixelBoundLow);
        //exitColumn = min(exitColumn, yPixelBoundHigh);
        tilesByColumn[j].push_back((float) exitColumn);
        //}
      }

    }
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
          result.push_back(Point2i(key, ii));
        }
      }
    }
  }


  Composite::Composite(StreamCam *parent) : update_mutex(new Poco::FastMutex()), parent(parent), root_offset(0.0, 0.0),
                                            max_offset(0.0, 0.0) {
    flat_field = parent->flat_field2X;

    //local_quality_score = score_image_2X(4852,6464,2190);
  }


  void CompositeVoronoi::update_Bbox_no_composite(std::vector<RegInfo> new_info) {
    bool update_box = false;

    //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.
    Vec2 temp_offset = root_offset;

    // If any new frames extend beyond the current extent, expand cv image dimensions
    for (int i = 0; i < new_info.size(); i++) {
      if (new_info[i].absoluteCoords.x < root_offset.x) {
        update_box = true;
        root_offset.x = new_info[i].absoluteCoords.x;
      }
      if (new_info[i].absoluteCoords.y < root_offset.y) {
        update_box = true;
        root_offset.y = new_info[i].absoluteCoords.y;
      }

      if (new_info[i].absoluteCoords.x + 6464 > max_offset.x) {
        update_box = true;
        max_offset.x = new_info[i].absoluteCoords.x + 6464;
      }
      if (new_info[i].absoluteCoords.y + 4852 > max_offset.y) {
        update_box = true;
        max_offset.y = new_info[i].absoluteCoords.y + 4852;
      }

    }
    if (update_box) {
      auto topLevelBeforeAdding = imagePyramid->level.back();
      unsigned int tile_size = imagePyramid->level[0]->getTileSize();
      unsigned int topLogicSize = imagePyramid->level.back()->getLogicSize();
      bool addedLevel = false;
      while (topLogicSize < max_offset.x - root_offset.x || topLogicSize < max_offset.y - root_offset.y) {
        addedLevel = true;
        unsigned int logic_size = 2 * topLogicSize;
        int levelWithinPyramid = imagePyramid->level.size();
        assert(pow(2, levelWithinPyramid) == logic_size / tile_size);
        imagePyramid->level.push_back(
            std::make_shared<TiledImage>(imagePyramid, tile_size, logic_size, levelWithinPyramid));
        topLogicSize = logic_size;
      }
      if (addedLevel) {
        auto tl = topLevelBeforeAdding->getIJ(Point2f(root_offset.x, root_offset.y));
        auto br = topLevelBeforeAdding->getIJ(Point2f(max_offset.x, max_offset.y));
        for (int x = tl.x; x <= br.x; x++) {
          for (int y = tl.y; y <= br.y; y++) {
            if (topLevelBeforeAdding->tiles(x, y) != nullptr) {
              auto tile = Point2i(x, y);
              auto myLevelRegion = cv::Rect_<float>(tile.x * tile_size, tile.y * tile_size, tile_size, tile_size);
              topLevelBeforeAdding->tileUpwards(tile, myLevelRegion, *topLevelBeforeAdding->tiles(tile.x, tile.y));
            }
          }
        }
      }
    }
  }


  void Composite::update_Bbox(std::vector<RegInfo> new_info) {

    bool update_box = false;

    //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.
    Vec2 temp_offset = root_offset;

    // If any new frames extend beyond the current extent, expand cv image dimensions
    for (int i = 0; i < new_info.size(); i++) {
      if (new_info[i].absoluteCoords.x < root_offset.x) {
        update_box = true;
        root_offset.x = new_info[i].absoluteCoords.x;
      }
      if (new_info[i].absoluteCoords.y < root_offset.y) {
        update_box = true;
        root_offset.y = new_info[i].absoluteCoords.y;
      }

      if (new_info[i].absoluteCoords.x + 6464 > max_offset.x) {
        update_box = true;
        max_offset.x = new_info[i].absoluteCoords.x + 6464;
      }
      if (new_info[i].absoluteCoords.y + 4852 > max_offset.y) {
        update_box = true;
        max_offset.y = new_info[i].absoluteCoords.y + 4852;
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
                 Size(composite.cols / pow(2, levelWithinPyramid), composite.rows / pow(2, levelWithinPyramid)));
          tiledImageBounds = cv::Rect_<float>(root_offset.x, root_offset.y, composite.cols, composite.rows);
          imagePyramid->level.back()->insertMat(temp, tiledImageBounds);
        }

      }
    }
  };

  void Composite::add_images(std::vector<RegInfo> new_info) {

    std::vector<unsigned long int> indexes;

    for (int i = 0; i < new_info.size(); i++) {
      indexes.push_back(new_info[i].index);
    }

    // get a copy of references to all images at once so that only one mutex lock is needed
    std::vector<Image *> images = parent->get_image_refs(indexes);

    //this may need to be placed inside the below for loop if frames ever vary in size. For now it is here so the mask only needs to be built once
    cv::Size image_size(images[0]->width, images[0]->height);
    /*
    Mat mask = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);

    circle(mask, cv::Point(image_size.width/2, image_size.height/2), 2190, cv::Scalar(255), -1);
    */

    for (int i = 0; i < images.size(); i++) {
      //calculate where the new image will be copied to in the composite
      Rect copyzone = Rect(new_info[i].absoluteCoords.x - root_offset.x,
                           new_info[i].absoluteCoords.y - root_offset.y, images[i]->width, images[i]->height);

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

  void Composite::update(std::vector<RegInfo> new_info) {
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

  void CompositeVoronoi::debug_write_contribution_on_grid(std::string name, pathCam::Vec2 absCoord, cv::Mat &img,
                                                          cv::Mat &mask) {
    int k = int(absCoord.x);
    k = 512 - k % 512;
    if (k < 0) { k = abs(k); }
    Mat debugmat = Mat::zeros(4852, 6464, CV_8UC4);
    img.copyTo(debugmat, mask);
    for (int m = k; m < debugmat.cols; m += 512) {
      cv::line(debugmat, cv::Point(m, 0), cv::Point(m, 4852), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(m + 1, 0), cv::Point(m + 1, 4852), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(m + 2, 0), cv::Point(m + 2, 4852), Scalar(0, 0, 255, 255));

    }
    k = int(absCoord.y);
    k = 512 - k % 512;

    if (k < 0) { k = abs(k); }
    for (int m = k; m < debugmat.rows; m += 512) {
      cv::line(debugmat, cv::Point(0, m), cv::Point(6464, m), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(0, m + 1), cv::Point(6464, m + 1), Scalar(0, 0, 255, 255));
      cv::line(debugmat, cv::Point(0, m + 2), cv::Point(6464, m + 2), Scalar(0, 0, 255, 255));

    }
    imwrite(name, debugmat);
  }


  void ImageToTileCopyRunnable::run() {

    auto composite = parent->composites[component_membership];
    try {
      auto mask = composite->polyMaskOutput;
      auto imageMat = composite->fourChannelPreallocated;
      auto tileSize = composite->imagePyramid->level[0]->getTileSize();
      auto tileBox = cv::Rect_<float>(tileSize * tile.x, tileSize * tile.y, tileSize, tileSize);
      auto imageBox = cv::Rect_<float>(image->absoluteCoords.x, image->absoluteCoords.y, image->width, image->height);

      composite->imagePyramid->level[0]->inserTileAtBase(imageMat, mask, imageBox, {tile});
    }
    catch (cv::Exception &e) {
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

    auto images = parent->get_image_refs(image_indexes);
    blurVals.clear();
    blurVals.resize(images.size());
    for (int i = 0; i < images.size(); i++) {
      blurVals[i] = images[i]->blurVariance;
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
        if (img->blurVariance > m - 1.0 * stdev) {
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


  void CompositeVoronoi::perform_global_alignment(unsigned int flag, double closenessFactor) {
    //flag == 0 will pull delaunay edges. Flag == 1 will pull all possible overlaps < closenessFactor
    if(delaunayMembers.size() < 10){
      return;
    }
    if (!needsAlignment) {
      return;
    }
    needsAlignment = false;

    if (flag == 1) {
      std::vector<std::tuple<unsigned long, unsigned long, int, bool>> indexIndexEdgenumJobneeded;
      int edgeNumber = 0;
      for (int i = 0; i < memberImages.size() - 1; i++) {
        for (int j = i + 1; j < memberImages.size(); j++) {
          auto image1 = memberImages[j].first;
          auto image2 = memberImages[i].first;

          if (abs(image2->absoluteCoords.x - image1->absoluteCoords.x) < closenessFactor * image_size.width &&
              abs(image2->absoluteCoords.y - image1->absoluteCoords.y) < closenessFactor * image_size.height) {
//            if (parent->matchM.match[image1->index][image2->index] != nullptr) {
//              indexIndexEdgenumJobneeded.push_back({image1->index, image2->index, edgeNumber, false});
//            } else {
            indexIndexEdgenumJobneeded.push_back({image1->index, image2->index, edgeNumber, true});
            //}
          }
        }
      }

      parent->resize_mmatch_mutex->readLock();
      matchedEdges.resize(indexIndexEdgenumJobneeded.size(), {-1, -1});

      for (int i = 0; i < indexIndexEdgenumJobneeded.size(); i++) {
        auto edge = indexIndexEdgenumJobneeded[i];
        auto idx1 = std::get<0>(edge);
        auto idx2 = std::get<1>(edge);

        if (std::get<3>(edge)) {
          auto sm = new SingleMatchRunnable(parent, idx1, idx2, componentIndex, i, 0);
          matchableCount++;
          parent->JobQ->add_runnable(sm);
        } else {
          matchedEdges[i].first = idx1;
          matchedEdges[i].second = idx2;
        }
      }
    } else if (flag == 0) {

      exclude_for_blur();
      //collect list of all edges.
      std::vector<Vec4f> edges;
      std::vector<Vec2i> verticePairs;
      std::vector<Point2f> coords;

      subdiv.getEdgeList(edges);
      matchedEdges.resize(edges.size(), {-1, -1}); //preallocated to avoid mutex

      //this mutex is locked here because each job created in the following loop needs this mutex to be read locked.
      //They will not cause a reallocation because these images have already had initial matches, meaning the match
      //matrix has already been resized to accommodate them. Its unlocked at end of loop
      parent->resize_mmatch_mutex->readLock();

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
    parent->resize_mmatch_mutex->unlock();


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
      if (matchedEdges[i].first != 0) {
        if (frameIndexToSystemIndex.find(matchedEdges[i].first) == frameIndexToSystemIndex.end()) {
          long val = frameIndexToSystemIndex.size();
          frameIndexToSystemIndex[matchedEdges[i].first] = val;
          systemIndexToFrameIndex[val] = matchedEdges[i].first;
        }
      }
      if (matchedEdges[i].second != 0) {
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

      auto verify1 = parent->reg_results[matchedEdges[i].first].absoluteCoords;
      auto verify2 = parent->reg_results[matchedEdges[i].second].absoluteCoords;
      auto val = abs(verify1.x - verify2.x - pwr->t_x);

      if (val > valtest1) {
        valtest1 = val;
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
      auto coords = parent->reg_results[systemIndexToFrameIndex[i]].absoluteCoords;
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

    coopers_conjugate_gradient(A, xpr, xac, 400, 0.0000001, false, systemIndexToFrameIndex, 0.05, ypr);
    coopers_conjugate_gradient(A, ypr, yac, 400, 0.0000001, false, systemIndexToFrameIndex, 0.05, xpr);

    auto testValAfter = norm(A * xac - xpr);
    int k = 0;
    /*
    //for debug, test for change in error
    double valtest2 = 0;
    std::vector<int> bins2;
    for (int i = 0; i < matchedEdges.size(); i++) {
      auto pwr = parent->matchM.match[matchedEdges[i].first][matchedEdges[i].second];
      Vec2 verify1;
      verify1.x = xac.at<double>(frameIndexToSystemIndex[matchedEdges[i].first]);
      verify1.y = yac.at<double>(frameIndexToSystemIndex[matchedEdges[i].first]);

      Vec2 verify2;
      verify2.x = xac.at<double>(frameIndexToSystemIndex[matchedEdges[i].second]);
      verify2.y = yac.at<double>(frameIndexToSystemIndex[matchedEdges[i].second]);
      double val = abs(verify1.x - verify2.x - pwr->t_x);
      if (val > valtest2) {
        valtest2 = val;
      }
      if ((int) val >= bins2.size()) {
        bins2.resize((int) val + 1, 0);
        bins2[(int) val] = 1;
      } else {
        bins2[(int) val]++;
      }

    }

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




    double valtest3 = 0;
    std::vector<int> bins3;
    for (int i = 0; i < matchedEdges.size(); i++) {
      auto pwr = parent->matchM.match[matchedEdges[i].first][matchedEdges[i].second];
      Vec2 verify1;
      verify1.x = xLP.at<double>(frameIndexToSystemIndex[matchedEdges[i].first]);
      verify1.y = yac.at<double>(frameIndexToSystemIndex[matchedEdges[i].first]);

      Vec2 verify2;
      verify2.x = xLP.at<double>(frameIndexToSystemIndex[matchedEdges[i].second]);
      verify2.y = yac.at<double>(frameIndexToSystemIndex[matchedEdges[i].second]);
      double val = abs(verify1.x - verify2.x - pwr->t_x);
      if (val > valtest3) {
        valtest3 = val;
      }

      if ((int) val >= bins3.size()) {
        bins3.resize((int) val + 1, 0);
        bins3[(int) val] = 1;
      } else {
        bins3[(int) val]++;
      }


    }
*/
    auto start = std::chrono::high_resolution_clock::now();

    self_reset();
    parent->update_observers();
    parent->reg_results_mutex->readLock();
    std::vector<RegInfo> newinfo;
    for (auto [i, elm]: systemIndexToFrameIndex) {
      parent->reg_results[systemIndexToFrameIndex[i]].absoluteCoords.x = xac.at<double>(i);
      parent->reg_results[systemIndexToFrameIndex[i]].absoluteCoords.y = yac.at<double>(i);
      newinfo.push_back(parent->reg_results[systemIndexToFrameIndex[i]]);
    }
    parent->reg_results_mutex->unlock();

    update(newinfo);
    //rebuild_DT_elementwise(newinfo);
    //create_and_submit_rebuild_jobs();


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;

  }


  void CompositeVoronoi::coopers_conjugate_gradient(cv::Mat A, cv::Mat b, cv::Mat x, int steps, double epsilon,
                                                    bool shouldCleanData,
                                                    std::map<long, long> &systemIndexToFrameIndex, double epsilonClean,
                                                    cv::Mat bOther) {
    std::vector<long> indexesRemoved;
    cv::Mat ATranspose = A.t();
    cv::Mat ATA = ATranspose * A;
    cv::Mat ATb = ATranspose * b;

    cv::Mat r = ATb - (ATA * x);
    cv::Mat p = r.clone();
    double n0 = cv::norm(A * x - b);
    double n1;
    for (int i = 0; i < steps; i++) {

      double rdr = r.dot(r);
      Mat ATAp = ATA * p;
      double stepSize = rdr / (p.dot(ATAp));

//      std::cout << "x before" << std::endl;
//      for (int i = 0; i < x.rows; i++) {
//        std::cout << std::to_string(x.at<double>(i)) << std::endl;
//      }
      x += stepSize * p;
//      std::cout << std::endl;

//      std::cout << "x after" << std::endl;
//      for (int i = 0; i < x.rows; i++) {
//        std::cout << std::to_string(x.at<double>(i)) << std::endl;
//      }
//      std::cout << std::endl;
      n1 = cv::norm(A * x - b);

      r -= stepSize * ATAp;
      double adjustment = r.dot(r) / rdr;
      p = r + adjustment * p;

      std::cout << "step " + std::to_string(i) + "   norm difference " + std::to_string(abs(n0 - n1)) +
                   "   norm of residual " + std::to_string(abs(n1)) << std::endl;

      if (shouldCleanData && abs(n0 - n1) < cv::norm(n1) * epsilonClean) {


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

