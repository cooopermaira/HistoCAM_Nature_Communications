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

    fPoint top_left = box.getTopLeft();
    fPoint bottom_right = box.getBottomRight();
    bottom_right.x -= 1;
    bottom_right.y -= 1;

    for (auto &tile: retileIndices) {
        int i = tile.getX();
        int j = tile.getY();


        if (tiles(i, j) == NULL) {
            cvTiles(i,j) = new Mat(tile_size,tile_size,CV_8UC4); //cvMat is already heap allocated
            tiles(i, j) = new juce::Image(juce::Image::PixelFormat::ARGB, tile_size, tile_size, true);
        }

        fRectangle tile_box = fRectangle(i * float(logic_size),
                                         j * float(logic_size),
                                         logic_size,
                                         logic_size);

        fRectangle image_box = tile_box.getIntersection(box);

        fPoint offset = box.getTopLeft();

        matToImage4Channel(image_in, tiles(i, j), offset * scale, image_box * scale, tile_box * scale);

    }
}



void TiledImage::tileUpwards(fRectangle baseLevelTile) {
    //find appropriate region of upper level
    float xloc = baseLevelTile.getX() / 2.f;
    float yloc = baseLevelTile.getY() / 2.f;
    float width = baseLevelTile.getWidth() / 2.f;
    float height = baseLevelTile.getHeight() / 2.f;
    auto region = fRectangle(xloc,yloc,width,height);

    //find appropriate tile in level up
    auto tileIndex = parent->level[levelWithinPyramid + 1]->getIJ(baseLevelTile.getCentre());

    //make region relative to tile
    float xlocUp = (long)region.getX() % tile_size;
    float ylocUp = (long)region.getY() % tile_size;

    //calculate ROI
    //cv::Rect ROI(baseLevelTile.getX(),baseLevelTile.getY(),);

    //resize self cv image into their cv image ROI
    iPoint myIndex = getIJ(baseLevelTile.getCentre());
    Mat* mine = cvTiles(myIndex.getX(),myIndex.getY());
    Mat* theirs = parent->level[levelWithinPyramid + 1]->cvTiles(tileIndex.getX(),tileIndex.getY());
    //resize(cvTiles(),parent->level[levelWithinPyramid + 1]->cvTiles(tileIndex.getX(),tileIndex.getY());

    //bitmap memcpy my cv image into juce image
    int k = 0;
}


void TiledImage::matToImage4Channel(const cv::Mat &mat, juce::Image *image, fPoint offset, fRectangle image_box,
                                    fRectangle tile_box) {
    cv::Rect ROIrect((int) (image_box.getX() - offset.getX()),
                     (int) (image_box.getY() - offset.getY()),
                     (int) image_box.getWidth(),
                     (int) image_box.getHeight());
    if(ROIrect.width * ROIrect.height > 0) {
        cv::Mat ROI = mat(ROIrect);
        const size_t numberOfBytesToCopy = 4 * ROI.cols;

        Image::BitmapData bitmap_data(*image, 0, 0, ROI.cols, ROI.rows, Image::BitmapData::ReadWriteMode::writeOnly);

        for (int row_index = 0; row_index < ROI.rows; row_index++) {
            auto *src_ptr = ROI.ptr(row_index);
            auto *dst_ptr = bitmap_data.getLinePointer(row_index + image_box.getY() - tile_box.getY());
            std::memcpy(dst_ptr, src_ptr, numberOfBytesToCopy);
        }

        //tileUpwards(tile_box);
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
            if(test > 0){
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
            if (tiles(i, j) == NULL) {
                tiles(i, j) = new juce::Image(juce::Image::PixelFormat::ARGB, tile_size, tile_size, true);
            }

            fRectangle tile_box = fRectangle(i * float(logic_size),
                                             j * float(logic_size),
                                             logic_size,
                                             logic_size);

            fRectangle image_box = tile_box.getIntersection(box);


            fPoint offset = box.getTopLeft();
            if(image_in.type() == CV_8UC3) {
                matToImage(image_in, tiles(i, j), offset * scale, image_box * scale, tile_box * scale);
            }else if(image_in.type() == CV_8UC4){
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
