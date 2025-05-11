//
// Created by cooper on 5/9/25.
//
#include "pathCam.h"

namespace pathCam {
#ifdef HAVE_OPENCV_CUDAARITHM

    void CompositeVoronoi::GPU_add_images_no_composite(std::vector<RegInfo *> new_info, bool _force_add) {
        // get a copy of references to all images at once so that only one mutex lock is needed
        std::vector<unsigned long> indexes;
        for (int i = 0; i < new_info.size(); i++) {
            indexes.push_back(new_info[i]->index);
        }
        std::vector<Image *> images = parent->get_image_refs(indexes);
        bool update = false;

        for (int i = 0; i < images.size(); i++) {

            //add point to delaunay triangulation
            std::vector<Point2i> face;
            auto fShift = Point2f(new_info[i]->absoluteCoords.x, new_info[i]->absoluteCoords.y);
            auto res = add_point_to_delaunay_triangulation(fShift, images[i], face, _force_add);


            update = true;
            images[i]->vertexId = res;

            //indicate that a new image has been added since last global alignment
            needsAlignment = true;

            //debayer image on gpu
            cuda::GpuMat image_Mat(image_size, CV_8U, images[i]->get_raw_cuda());
            cuda::cvtColor(image_Mat, threeChannelPrealGPU, COLOR_BayerBG2BGR);

            if (componentMagLabel != 0) {
                //flatfield correct
                threeChannelPrealGPU.convertTo(convertHoldingGPU, CV_32F);
                cuda::divide(convertHoldingGPU, ffGPU, convertHoldingGPU, 1, CV_32F);
                //brighten
                cuda::pow(convertHoldingGPU, 1.1, convertHoldingGPU);
                convertHoldingGPU.convertTo(threeChannelPrealGPU, CV_8UC3);
            }

            // Mat test2 = Mat::zeros(image_size, CV_8UC3);
            // threeChannelPrealGPU.download(test2);
            // imwrite("/media/max/Data/testAfterPow.png",test2);
            //
            // Mat test(image_size,CV_8U);
            // image_Mat.download(test);
            // cvtColor(test,test,COLOR_BayerBG2BGR);
            // test.convertTo(test, CV_32F);
            //
            // Mat ffdownload(image_size, CV_32FC3);
            // ffGPU.download(ffdownload);
            //
            // Mat test1;
            // divide(test,ffdownload,test1,1,CV_32F);
            // cv::pow(test1,1.1,test1);
            // test1.convertTo(test1, CV_8U);
            // imwrite("/media/max/Data/testcpuDownload.png",test1);
            //
            // std::string filename = parent->get_flatfield(componentMagLabel);
            // size_t nBytes = parent->image_height * parent->image_width;
            // char *buffer = new char[nBytes];
            // std::ifstream stream;
            // stream.open(filename, std::ios::binary);
            // stream.read(buffer, nBytes);
            //
            // Mat fflocal(image_size,CV_8U,buffer);
            // cvtColor(fflocal,fflocal,COLOR_BayerBG2BGR);
            // fflocal.convertTo(fflocal, CV_32F);
            // auto scale = 1/170.0;
            // fflocal *= scale;
            //
            // Mat test3;
            // divide(test,fflocal,test3,1,CV_32F);
            // cv::pow(test3,1.1,test3);
            // test3.convertTo(test3, CV_8U);
            // imwrite("/media/max/Data/testcpuLoad.png",test3);


            //channelsGPU[0] = threeChannelPrealGPU; //3 channel
            cuda::split(threeChannelPrealGPU, channelsGPU);

            //add alpha channel
            channelsGPU.push_back(rectMaskGPU);
            cuda::merge(channelsGPU, fourChannelPrealGPU);

            //download back to cpu
            //fourChannelPreallocated.create(fourChannelPrealGPU.size(), fourChannelPrealGPU.type());
            fourChannelPrealGPU.download(fourChannelPreallocated);
            //imwrite("/media/max/Data/testAfterAlpha.png",fourChannelPreallocated);
            // char *rcv;
            // size_t nBytes = image_size.width * image_size.height * 4;
            // rcv = (char*)malloc(nBytes);
            // cudaMemcpy(rcv,fourChannelPrealGPU.data,  nBytes, cudaMemcpyDeviceToHost);
            // fourChannelPreallocated = Mat(image_size,CV_8UC4,rcv);
            //
            // imwrite("/media/max/Data/test2.png", fourChannelPreallocated);

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
                imagePyramid->insertTilesAtBase(fourChannelPreallocated, Mat(), imageBox, effectedTilesNoMask);
            }

            imagePyramid->insertTilesAtBase(fourChannelPreallocated, polyMaskOutput, imageBox, effectedTiles);

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

#endif
}
