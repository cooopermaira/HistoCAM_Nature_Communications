//
//  TiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#include "JuceHeader.h"


void TiledImage::insertMatAtBase(cv::Mat image_in, fRectangle box, std::vector<iPoint> retileIndices) {
    unsigned int width = image_in.cols;
    unsigned int height = image_in.rows;
    float scale = ((float) tile_size / (float) logic_size);

    jassert(int(box.getWidth()*scale) == width &&
                    int(box.getHeight()*scale) == height);

    bounds = bounds.getUnion(box);

    for (auto &tile: retileIndices) {
        int x = tile.getX();
        int y = tile.getY();

        makeTile(x, y);

        fRectangle tile_box = fRectangle(x * float(logic_size),
                                         y * float(logic_size),
                                         logic_size,
                                         logic_size);

        fRectangle image_box = tile_box.getIntersection(box);

        fPoint offset = box.getTopLeft(); //equivalent of root offset

        matToImage4Channel(image_in, x, y, offset * scale, image_box * scale, tile_box * scale);

    }
}


void TiledImage::matToImage4Channel(const cv::Mat &mat, int x, int y, fPoint rootOffset, fRectangle image_box,
                                    fRectangle tile_box) {
    cv::Rect ROIrect((int) (image_box.getX() - rootOffset.getX()),
                     (int) (image_box.getY() - rootOffset.getY()),
                     (int) image_box.getWidth(),
                     (int) image_box.getHeight());
    if (ROIrect.width * ROIrect.height > 0) {
        cv::Mat matROI = mat(ROIrect);
        const size_t numberOfBytesToCopy = 4 * matROI.cols;

        Image::BitmapData bitmap_data(*tiles(x, y), 0,
                                      0, matROI.cols, matROI.rows,
                                      Image::BitmapData::ReadWriteMode::writeOnly);

        for (int row_index = 0; row_index < matROI.rows; row_index++) {
            auto *src_ptr = matROI.ptr(row_index);
            auto *dst_ptr = bitmap_data.getPixelPointer(image_box.getX() - tile_box.getX(), row_index+image_box.getY() - tile_box.getY());
            std::memcpy(dst_ptr, src_ptr, numberOfBytesToCopy);
        }


        //prepare to tile upward
        auto tileROI = cv::Rect(image_box.getX() - tile_box.getX(), image_box.getY() - tile_box.getY(), matROI.cols, matROI.rows);

        //weird edge case
        Mat temp;
        if(matROI.cols < tile_size || matROI.rows < tile_size){
            temp = Mat::zeros(tile_size,tile_size,CV_8UC4);
            matROI.copyTo(temp(tileROI));
        }else{
            temp = matROI;
        }

        tileUpwards(iPoint(x,y),tile_box, temp);
    }
}


void TiledImage::tileUpwards(iPoint myTileIndex, fRectangle myLevelRegion, const cv::Mat &myCV) {
    if(myLevelRegion.getX() < 0 || myLevelRegion.getY() < 0){
        int k = 0;
    }
    //find appropriate region of upper level
    float xloc = myLevelRegion.getX() / 2.f;
    float yloc = myLevelRegion.getY() / 2.f;
    float width = myLevelRegion.getWidth() / 2.f;
    float height = myLevelRegion.getHeight() / 2.f;
    auto theirLevelRegion = fRectangle(xloc, yloc, width, height);

    //find appropriate tiles
    iPoint theirTileIndex;
    theirTileIndex.x = myTileIndex.getX() == -1 ? myTileIndex.getX() : myTileIndex.getX() / 2;
    theirTileIndex.y = myTileIndex.getY() == -1 ? myTileIndex.getY() : myTileIndex.getY() / 2;

    //instantiate
    parent->level[levelWithinPyramid + 1]->makeTile(theirTileIndex.getX(), theirTileIndex.getY());

    //make theirLevelRegion relative to tile
    float theirTileRegionX = (long) theirLevelRegion.getX() % tile_size;
    theirTileRegionX = theirTileRegionX < 0 ? theirTileRegionX + tile_size : theirTileRegionX;
    float theirTileRegionY = (long) theirLevelRegion.getY() % tile_size;
    theirTileRegionY = theirTileRegionY < 0 ? theirTileRegionY + tile_size : theirTileRegionY;

    //calculate theirROI
    cv::Rect theirROI(theirTileRegionX, theirTileRegionY, theirLevelRegion.getWidth(), theirLevelRegion.getHeight());

    //resize self cv image into their cv image ROI
    jassert(holdingMatrix.rows == myCV.rows / 2 && holdingMatrix.cols == myCV.cols / 2);
    resize(myCV,holdingMatrix,Size(holdingMatrix.cols,holdingMatrix.rows));


    //bitmap memcpy my cv image into juce image
    auto theirJuceImage = *parent->level[levelWithinPyramid + 1]->tiles(theirTileIndex.getX(), theirTileIndex.getY());
    auto bitmap_data = new Image::BitmapData(theirJuceImage, theirROI.x, theirROI.y, theirROI.width, theirROI.height,
                                  Image::BitmapData::ReadWriteMode::writeOnly);

    size_t bytesToCopy = 4 * theirROI.width;
    for (int row_index = 0; row_index < theirROI.height; row_index++) {
        auto *src_ptr = holdingMatrix.ptr(row_index);
        auto *dst_ptr = bitmap_data->getLinePointer(row_index);
        std::memcpy(dst_ptr, src_ptr, bytesToCopy);
    }

    delete bitmap_data;

    //continue up pyramid
    if (levelWithinPyramid + 1 < parent->level.size() - 1) {
        parent->level[levelWithinPyramid + 1]->tileUpwards(theirTileIndex,theirLevelRegion, holdingMatrix);
    }
}


void TiledImage::matToImage2(const cv::Mat &mat, juce::Image *image,
                             fPoint offset, fRectangle image_box,
                             fRectangle tile_box) {
    jassert(mat.type() == CV_8UC4);

    cv::Rect ROIrect((int) (image_box.getX() - offset.getX()),
                     (int) (image_box.getY() - offset.getY()),
                     (int) image_box.getWidth(),
                     (int) image_box.getHeight());

    cv::Mat ROI = mat(ROIrect);

    juce::Image::BitmapData data(*image, juce::Image::BitmapData::readWrite);

    for (int y = 0, v = (int) (image_box.getY() - tile_box.getY()); y < ROI.rows; ++y, ++v) {
        const auto *matRowPtr = ROI.ptr<cv::Vec4b>(y);
        for (int x = 0, u = (int) (image_box.getX() - tile_box.getX()); x < ROI.cols; ++x, ++u) {
            const cv::Vec4b &bgr = matRowPtr[x];
            jassert(u < (tile_size) and v < (tile_size));
            uint8 alpha = (bgr[2] == 0 && bgr[1] == 0 && bgr[0] == 0) ? 0 : 255;
            juce::Colour orig = data.getPixelColour(u, v);
            juce::Colour newColor = juce::Colour(bgr[2], bgr[1], bgr[0], alpha);
            auto test = newColor.getARGB();
            if (test > 0) {
                int k = 0;
            }
            if (alpha == 0) {
                data.setPixelColour(u, v, orig);
            } else {
                data.setPixelColour(u, v, juce::Colour(bgr[2], bgr[1], bgr[0], alpha));
            }
        }
    }

}


void TiledImage::matToImage(const cv::Mat &mat, juce::Image *image,
                            fPoint offset, fRectangle image_box,
                            fRectangle tile_box) {
    jassert(mat.type() == CV_8UC3);

    cv::Rect ROIrect((int) (image_box.getX() - offset.getX()),
                     (int) (image_box.getY() - offset.getY()),
                     (int) image_box.getWidth(),
                     (int) image_box.getHeight());

    cv::Mat ROI = mat(ROIrect);

    juce::Image::BitmapData data(*image, juce::Image::BitmapData::readWrite);

    for (int y = 0, v = (int) (image_box.getY() - tile_box.getY()); y < ROI.rows; ++y, ++v) {
        const auto *matRowPtr = ROI.ptr<cv::Vec3b>(y);
        for (int x = 0, u = (int) (image_box.getX() - tile_box.getX()); x < ROI.cols; ++x, ++u) {
            const cv::Vec3b &bgr = matRowPtr[x];
            jassert(u < (tile_size) and v < (tile_size));
            uint8 alpha = (bgr[2] == 0 && bgr[1] == 0 && bgr[0] == 0) ? 0 : 255;
            juce::Colour orig = data.getPixelColour(u, v);
            if (alpha == 0) {
                data.setPixelColour(u, v, orig);
            } else {
                data.setPixelColour(u, v, juce::Colour(bgr[2], bgr[1], bgr[0], alpha));
            }
        }
    }

}

void TiledImage::insertMat(cv::Mat image_in, fRectangle box) {
    unsigned int width = image_in.cols;
    unsigned int height = image_in.rows;
    float scale = ((float) tile_size / (float) logic_size);

    jassert(int(box.getWidth()*scale) == width &&
                    int(box.getHeight()*scale) == height);

    bounds = bounds.getUnion(box);

    fPoint top_left = box.getTopLeft();
    fPoint bottom_right = box.getBottomRight();
    bottom_right.x -= 1;
    bottom_right.y -= 1;


    for (int i = getIJ(top_left).getX(); i <= getIJ(bottom_right).getX(); i++) {
        for (int j = getIJ(top_left).getY(); j <= getIJ(bottom_right).getY(); j++) {

            makeTile(i, j);

            fRectangle tile_box = fRectangle(i * float(logic_size),
                                             j * float(logic_size),
                                             logic_size,
                                             logic_size);

            fRectangle image_box = tile_box.getIntersection(box);

            fPoint offset = box.getTopLeft();
            if (image_in.type() == CV_8UC3) {
                matToImage(image_in, tiles(i, j), offset * scale, image_box * scale, tile_box * scale);
            } else if (image_in.type() == CV_8UC4) {
                matToImage2(image_in, tiles(i, j), offset * scale, image_box * scale, tile_box * scale);
            }
        }
    }
}

std::vector<TileQuery> TiledImage::getTiles(fRectangle box) {
    std::vector<TileQuery> box_tiles;

    fPoint top_left = box.getTopLeft();
    fPoint bottom_right = box.getBottomRight();
    bottom_right.x--;
    bottom_right.y--;


    for (int i = getIJ(top_left).getX(); i <= getIJ(bottom_right).getX(); i++) {
        for (int j = getIJ(top_left).getY(); j <= getIJ(bottom_right).getY(); j++) {
            int x = i * (int) logic_size - box.getX();
            int y = j * (int) logic_size - box.getY();
            juce::Rectangle<float> rect = juce::Rectangle<float>(x, y, logic_size, logic_size);
            box_tiles.push_back(TileQuery(tiles(i, j), i, j, rect));
        }
    }

    return box_tiles;
}

void TiledImage::makeTile(int x, int y) {
    if (tiles(x, y) == NULL) {
        //cvTiles(x, y) = new Mat(Size(tile_size, tile_size), CV_8UC4);
        tiles(x, y) = new juce::Image(juce::Image::PixelFormat::ARGB, tile_size, tile_size, true);
    }
}