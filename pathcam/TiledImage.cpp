//
//  TiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#include "pathCam.h"

template<typename T>
cv::Rect_<T> rect_mult(cv::Rect_<T> r, T s) {
  return cv::Rect_<T>(r.x * s, r.y * s, r.width * s, r.height * s);
}

void TiledImage::inserTileAtBase(cv::Mat image_in, cv::Mat mask, cv::Rect_<float> box,
                                 std::vector<Point2i> retileIndices) {
  unsigned int width = image_in.cols;
  unsigned int height = image_in.rows;

  float scale = ((float) tile_size / (float) logic_size);

  assert(int(box.width*scale) == width &&
             int(box.height*scale) == height);

  bounds = bounds | box;

  for (auto &tile: retileIndices) {
    int x = tile.x;
    int y = tile.y;

    makeTile(x, y);

    cv::Rect_<float> tile_box = cv::Rect_<float>(x * float(logic_size),
                                                 y * float(logic_size),
                                                 logic_size,
                                                 logic_size);

    cv::Rect_<float> image_box = tile_box & box;

    Point2f offset = box.tl(); //equivalent of root offset

    matToTile(image_in, mask, x, y, offset * scale,
              rect_mult<float>(image_box, scale),
              rect_mult<float>(tile_box, scale));

  }
}

void TiledImage::insertMatAtBase(cv::Mat image_in, cv::Rect_<float> box, std::vector<Point2i> retileIndices) {
  unsigned int width = image_in.cols;
  unsigned int height = image_in.rows;
  float scale = ((float) tile_size / (float) logic_size);

  assert(int(box.width*scale) == width &&
             int(box.height*scale) == height);

  bounds = bounds | box;

  for (auto &tile: retileIndices) {
    int x = tile.x;
    int y = tile.y;

    makeTile(x, y);

    cv::Rect_<float> tile_box = cv::Rect_<float>(x * float(logic_size),
                                                 y * float(logic_size),
                                                 logic_size,
                                                 logic_size);

    cv::Rect_<float> image_box = tile_box & box;

    Point2f offset = box.tl(); //equivalent of root offset

    matToImage4Channel(image_in, x, y, offset * scale,
                       rect_mult<float>(image_box, scale),
                       rect_mult<float>(tile_box, scale));

  }
}


void TiledImage::matToImage4Channel(const cv::Mat &mat, int x, int y, Point2f rootOffset, cv::Rect_<float> image_box,
                                    cv::Rect_<float> tile_box) {
  cv::Rect ROIrect((int) (image_box.x - rootOffset.x),
                   (int) (image_box.y - rootOffset.y),
                   (int) image_box.width,
                   (int) image_box.height);
  if (ROIrect.width * ROIrect.height > 0) {


    cv::Mat matROI = mat(ROIrect);
    auto tile = tiles(x, y);
    //prepare to tile upward
    auto tileROI = cv::Rect(image_box.x - tile_box.x, image_box.y - tile_box.y, matROI.cols,
                            matROI.rows);
    matROI.copyTo(tile(tileROI));
/*


    //weird edge case
    Mat temp;
    if (matROI.cols < tile_size || matROI.rows < tile_size) {
      temp = Mat::zeros(tile_size, tile_size, CV_8UC4);
      matROI.copyTo(temp(tileROI));
    } else {
      temp = matROI;
    }
*/
    tileUpwards(Point2i(x, y), tile_box, tile);
  }
}

void TiledImage::matToTile(const cv::Mat &mat, const cv::Mat &mask, int x, int y, cv::Point2f rootOffset,
                           cv::Rect_<float> image_box, cv::Rect_<float> tile_box) {
  cv::Rect ROIrect((int) (image_box.x - rootOffset.x),
                   (int) (image_box.y - rootOffset.y),
                   (int) image_box.width,
                   (int) image_box.height);
  if (ROIrect.width * ROIrect.height > 0) {
    cv::Mat matROI = mat(ROIrect);
    auto temp = tiles(x, y);
    auto tileROI = cv::Rect(image_box.x - tile_box.x, image_box.y - tile_box.y, matROI.cols,
                            matROI.rows);
    matROI.copyTo(temp(tileROI), mask(ROIrect));

    assert(tiles(x, y).rows == 512 && tiles(x, y).cols == 512);
    //prepare to tile upward


    //weird edge case
    /*
    Mat temp;
    if (matROI.cols < tile_size || matROI.rows < tile_size) {
      //temp = Mat::zeros(tile_size, tile_size, CV_8UC4);
      matROI.copyTo(temp(tileROI),mask(tileROI));
    } else {
      temp = matROI;
    }
*/
    tileUpwards(Point2i(x, y), tile_box, tiles(x, y));
  }
}


void TiledImage::tileUpwards(Point2i myTileIndex, cv::Rect_<float> myLevelRegion, const cv::Mat &myCV) {

  //find appropriate region of upper level
  float xloc = myLevelRegion.x / 2.f;
  float yloc = myLevelRegion.y / 2.f;
  float width = myLevelRegion.width / 2.f;
  float height = myLevelRegion.height / 2.f;
  auto theirLevelRegion = cv::Rect_<float>(xloc, yloc, width, height);

  //find appropriate tiles
  Point2i theirTileIndex;
  theirTileIndex.x = myTileIndex.x < 0 ? (myTileIndex.x - 1) / 2 : myTileIndex.x / 2;
  theirTileIndex.y = myTileIndex.y < 0 ? (myTileIndex.y - 1) / 2 : myTileIndex.y / 2;

  //instantiate
  parent->level[levelWithinPyramid + 1]->makeTile(theirTileIndex.x, theirTileIndex.y);

  //make theirLevelRegion relative to tile
  float theirTileRegionX = (long) theirLevelRegion.x % tile_size;
  theirTileRegionX = theirTileRegionX < 0 ? theirTileRegionX + tile_size : theirTileRegionX;
  float theirTileRegionY = (long) theirLevelRegion.y % tile_size;
  theirTileRegionY = theirTileRegionY < 0 ? theirTileRegionY + tile_size : theirTileRegionY;

  //calculate theirROI and grab tile
  cv::Rect theirROI(theirTileRegionX, theirTileRegionY, theirLevelRegion.width, theirLevelRegion.height);
  auto theirImage = parent->level[levelWithinPyramid + 1]->tiles(theirTileIndex.x, theirTileIndex.y);

  //resize self cv image into their cv image ROI
  assert(theirImage(theirROI).rows == myCV.rows / 2 && theirImage(theirROI).cols == myCV.cols / 2);
  resize(myCV, theirImage(theirROI), Size(theirImage(theirROI).cols, theirImage(theirROI).rows));


  //continue up pyramid
  if (levelWithinPyramid + 1 < parent->level.size() - 1) {
    parent->level[levelWithinPyramid + 1]->tileUpwards(theirTileIndex, theirLevelRegion, theirImage(theirROI));
  }
}


void TiledImage::matToImage2(const cv::Mat mat, cv::Mat image,
                             Point2f offset, cv::Rect_<float> image_box,
                             cv::Rect_<float> tile_box) {

  assert(mat.type() == CV_8UC4);

  cv::Rect ROIrect((int) (image_box.x - offset.x),
                   (int) (image_box.y - offset.y),
                   (int) image_box.width,
                   (int) image_box.height);

  cv::Mat ROI = mat(ROIrect);
  if (ROI.cols * ROI.rows > 0) {

    ROI.copyTo(image(cv::Rect(image_box.x - tile_box.x,
                              image_box.y - tile_box.y,
                              ROI.cols,
                              ROI.rows)));
  }

}

void TiledImage::resetEdges(Point2i topLeft, Point2i bottomRight) {
  auto tL = getIJ(topLeft);
  auto bR = getIJ(bottomRight);
  std::vector<Point2i> edgeTiles;
  bool escape = false;
  for (int x = tL.x; x <= bR.x; x++) {
    for (int y = tL.y; y <= bR.y; y++) {
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          if (!tiles(x + i, y + j).data) {
            edgeTiles.push_back(Point2i(x, y));
            escape = true;
            break;
          }
        }
        if (escape) {
          escape = false;
          break;
        }
      }
    }
  }
  for (int i = 0; i < edgeTiles.size(); i++) {
    tiles(edgeTiles[i].x, edgeTiles[i].y).release();
  }
}


void TiledImage::matToImage(const cv::Mat &mat, cv::Mat *image,
                            Point2f offset, cv::Rect_<float> image_box,
                            cv::Rect_<float> tile_box) {

  assert(mat.type() == CV_8UC3);

  cv::Rect ROIrect((int) (image_box.x - offset.x),
                   (int) (image_box.y - offset.y),
                   (int) image_box.width,
                   (int) image_box.height);

  cv::Mat ROI = mat(ROIrect);
  ROI.copyTo((*image)(cv::Rect(image_box.x - tile_box.x,
                               image_box.y - tile_box.y,
                               ROI.cols,
                               ROI.rows)));

}

void TiledImage::insertMat(cv::Mat image_in, cv::Rect_<float> box) {

  unsigned int width = image_in.cols;
  unsigned int height = image_in.rows;
  float scale = ((float) tile_size / (float) logic_size);

  assert(int(box.width*scale) == width &&
             int(box.height*scale) == height);

  bounds = bounds | box;

  Point2f top_left = box.tl();
  Point2f bottom_right = box.br();
  bottom_right.x -= 1;
  bottom_right.y -= 1;


  for (int i = getIJ(top_left).x; i <= getIJ(bottom_right).x; i++) {
    for (int j = getIJ(top_left).y; j <= getIJ(bottom_right).y; j++) {

      makeTile(i, j);

      cv::Rect_<float> tile_box = cv::Rect_<float>(i * float(logic_size),
                                                   j * float(logic_size),
                                                   logic_size,
                                                   logic_size);

      cv::Rect_<float> image_box = tile_box & box;

      Point2f offset = box.tl();

      if (image_in.type() == CV_8UC4) {
        matToImage2(image_in, tiles(i, j), offset * scale,
                    rect_mult<float>(image_box, scale),
                    rect_mult<float>(tile_box, scale));
      } else {
        throw std::invalid_argument("received wrong channel of matrix");
      }
    }
  }

}

std::vector<TileQuery> TiledImage::getTiles(cv::Rect_<float> box) {
  std::vector<TileQuery> box_tiles;

  Point2f top_left = box.tl();
  Point2f bottom_right = box.br();
  bottom_right.x--;
  bottom_right.y--;


  for (int i = getIJ(top_left).x; i <= getIJ(bottom_right).x; i++) {
    for (int j = getIJ(top_left).y; j <= getIJ(bottom_right).y; j++) {
      int x = i * (int) logic_size - box.x;
      int y = j * (int) logic_size - box.y;
      cv::Rect_<float> rect = cv::Rect_<float>(x, y, logic_size, logic_size);
      box_tiles.push_back(TileQuery(tiles(i, j), i, j, rect));
    }
  }

  return box_tiles;
}

void TiledImage::makeTile(int x, int y) {
  if (x < -1000 || x > 1000 || y < -1000 || y > 1000) {
    int k = 0;
  }
  if (!tiles(x, y).data) {
    //tilesWithData.push_back(Point2i(x, y));
    tiles(x, y) = Mat::zeros(Size(tile_size, tile_size), CV_8UC4);
    //tiles(x, y) = new juce::Image(juce::Image::PixelFormat::ARGB, tile_size, tile_size, true);
  }
}
