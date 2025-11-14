//
// Created by cooper maira on 11/11/25.
//

#include "pathCam.h"

namespace pathCam {
    MetricComposite::MetricComposite(StreamCam *parent, Size image_size, int _componentIndex) : Composite(
        parent, image_size, _componentIndex) {
        frameDelay = 10;
        waitingFrames.resize(frameDelay, {nullptr, {}});
        compositeImage = imagePyramid->level[0];

        rectMask = Mat(image_size, CV_8U, cv::Scalar(255));
        rectMaskGPU = cuda::GpuMat(image_size,CV_8UC1, rectMask.data);
        threeChannelPreallocated = Mat(image_size, CV_8UC3);
        threeChannelPrealGPU = cuda::GpuMat(image_size, CV_8UC3, threeChannelPreallocated.data);
        fourChannelPreallocated = Mat::zeros(image_size, CV_8UC4);
        fourChannelPrealGPU = cuda::GpuMat(image_size,CV_8UC4, fourChannelPreallocated.data);

        circleMask = Mat::zeros(image_size, CV_8U);
        circle(circleMask, cv::Point(image_size.width / 2, image_size.height / 2), parent->scope_radius,
               Scalar(255),
               -1);
    }

    void MetricComposite::update() {
        //make sure component is placed in MR image
        if (imagePyramid->scale == 0) {
            assert(!staging.empty());
            establish_scale_at_root(staging.front()->image);
            assert(imagePyramid->scale > 0);
        }


        if (!staging.empty()) {
            auto ri = staging.front();
            auto img = ri->image;
            staging.pop();
            update_Bbox_no_composite({ri});

            //grab affected tiles with their category of coverage
            auto affectedPyramidTilesWithStatus = calculate_affected_tiles_with_status(
                Point2f(ri->absoluteCoords.x, ri->absoluteCoords.y));
            std::vector<Point2i> immediateProcessingTiles;

            waitingFrames[positionForNextWaitngFrame % frameDelay] = {img, {}};

            //find what tiles raise status category of pyramid tiles
            for (auto &el: affectedPyramidTilesWithStatus) {
                auto &pyramidTileObj = compositeImage->getTile(el.first.x, el.first.y);
                if (pyramidTileObj.status < el.second) {
                    //this frame improves status of this pyramid tile and should fill the tile without delay
                    if (el.second == TileObj::singleFrameCoverage) {
                        pyramidTileObj.owner = img;
                        pyramidTileObj.motionBlur = img->motionBlur;
                    }
                    pyramidTileObj.status = el.second;
                    immediateProcessingTiles.push_back(el.first);
                } else if (el.second == TileObj::singleFrameCoverage && pyramidTileObj.owner->motionBlur > img->
                           motionBlur) {
                    waitingFrames[positionForNextWaitngFrame % frameDelay].second.push_back(el.first);
                }
            }
            ++positionForNextWaitngFrame;

            if (!immediateProcessingTiles.empty()) {
                process_tiles(img, immediateProcessingTiles);
                ++debugFrameCount;
                debugTileCount1 += immediateProcessingTiles.size();
            }

            if (imagePyramid->scale > 0) {
                float x = (imagePyramid->offset.x + img->absoluteCoords.x) * imagePyramid->scale;
                float y = (imagePyramid->offset.y + img->absoluteCoords.y) * imagePyramid->scale;
                float w = parent->image_width * imagePyramid->scale;
                float h = parent->image_height * imagePyramid->scale;
                bool showAsCircle = (componentMagLabel == Image::_2X);

                parent->update_last_frame(Rect_<float>(x, y, w, h), showAsCircle, componentIndex,
                                          Image::get_label(componentMagLabel));
            }
        } else {
            waitingFrames[positionForNextWaitngFrame % frameDelay] = {nullptr, {}};
            ++positionForNextWaitngFrame;
        }


        //process delayed frames, allowing them to blur correct if necessary
        //this is an erase-remove_if implementation with a lambda function inside that updates tileObj if img should be owner,
        //otherwise it removes the tile from the img's list
        for (auto &imgTilesetPair: waitingFrames) {
            if (!imgTilesetPair.first){continue;}
            imgTilesetPair.second.erase(
                std::remove_if(imgTilesetPair.second.begin(),
                               imgTilesetPair.second.end(),
                               [&](Point2i &tile) {
                                   auto &tileObj = compositeImage->getTile(tile.x, tile.y);
                                   if (tileObj.owner == imgTilesetPair.first) {
                                       return false;
                                   }
                                   if (tileObj.motionBlur > imgTilesetPair.first->motionBlur) {
                                       tileObj.owner = imgTilesetPair.first;
                                       tileObj.motionBlur = imgTilesetPair.first->motionBlur;
                                       return false;
                                   }
                                   return true;
                               }), imgTilesetPair.second.end()
            );
        }

        //once delay is met, process frames
        auto imgTileSet = waitingFrames[positionForNextWaitngFrame % frameDelay];
        if (positionForNextWaitngFrame > frameDelay && !imgTileSet.second.empty()) {
            process_tiles(imgTileSet.first,imgTileSet.second);
            debugTileCount2 += imgTileSet.second.size();
            ++debugFrameCount;
        }
    }

    void MetricComposite::process_tiles(Image *img, std::vector<Point2i> &tiles) {
        assert(parent->unifiedMemory); //change this to a fix later
        if (!img->in_memory()) {
            img->load_raw_from_disk();
        }

        //process status improvement frames
        prepare_4CPA(img, tiles);

        //calculate region of pyramid for data placement
        auto imageBox = cv::Rect_<float>(img->absoluteCoords.x, img->absoluteCoords.y, img->width, img->height);
        Mat mask = componentMagLabel == Image::_2X ? circleMask : rectMask;

        imagePyramid->insertTilesAtBase(fourChannelPreallocated, mask, imageBox, tiles);
    }


    std::vector<std::pair<Point2i, int> > MetricComposite::calculate_affected_tiles_with_status(Point2f AbC) {
        std::vector<std::pair<Point2i, int> > results;

        if (componentMagLabel == Image::_2X) {
            Point2i center(imageSize.width / 2 + AbC.x, imageSize.height / 2 + AbC.y);

            int distance = parent->scope_radius + parent->tileSize;
            int distSq = pow(parent->scope_radius, 2);

            Point2f ulP(center.x - distance, center.y - distance);
            Point2f lrP(center.x + distance, center.y + distance);

            auto ulTileInd = imagePyramid->level[0]->getIJ(ulP);
            auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

            int inlierCorners;
            for (int x = ulTileInd.x; x <= lrTileInd.x; ++x) {
                for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {
                    inlierCorners = 0;
                    for (int i = 0; i < 4; ++i) {
                        int locX = parent->tileSize * (x + i % 2);
                        int locy = parent->tileSize * (y + i / 2);
                        if (pow(locX - center.x, 2) +
                            pow(locy - center.y, 2) < distSq) {
                            ++inlierCorners;
                        }
                    }
                    if (inlierCorners == 4) {
                        results.emplace_back(Point2i(x, y), TileObj::singleFrameCoverage);
                    } else if (inlierCorners > 0) {
                        results.emplace_back(Point2i(x, y), TileObj::partialCoverage);
                    }
                }
            }
        } else {
            auto lrP = AbC + Point2f(float(imageSize.width), float(imageSize.height));

            auto ulTileInd = imagePyramid->level[0]->getIJ(AbC);
            auto lrTileInd = imagePyramid->level[0]->getIJ(lrP);

            for (int x = ulTileInd.x; x <= lrTileInd.x; ++x) {
                for (int y = ulTileInd.y; y <= lrTileInd.y; ++y) {
                    if (parent->tileSize * x >= AbC.x && parent->tileSize * x + parent->tileSize <= AbC.x + imageSize.
                        width &&
                        parent->tileSize * y >= AbC.y && parent->tileSize * y + parent->tileSize <= AbC.y + imageSize.
                        height) {
                        results.emplace_back(Point2i(x, y), TileObj::singleFrameCoverage);
                    } else {
                        results.emplace_back(Point2i(x, y), TileObj::partialCoverage);
                    }
                }
            }
        }
        return results;
    }
}
