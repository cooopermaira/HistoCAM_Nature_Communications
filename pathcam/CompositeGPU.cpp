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
            if (_face[i].x == _face[i-1].x && _face[i].y == _face[i-1].y) {
                _face.erase(_face.begin() + i);
            }else {
                i++;
            }
        }
        //ensure_clockwise(_face);
    }


    void CompositeVoronoi::ensure_clockwise(std::vector<Point2i> &_face) {
        double area = 0.0;
        for (int i = 1; i < _face.size(); ++i) {
            const cv::Point2i& p0 = _face[i - 1];
            const cv::Point2i& p1 = _face[i];
            area += (p0.x * p1.y - p1.x * p0.y);
        }
        if(area < 0){
            std::reverse(_face.begin(), _face.end());
        }
        else {
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

        Mat X,Y;
        repeat(x_row,image_size.height,1,X);
        repeat(y_col,1,image_size.width,Y);

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
        for (int i = 1;i<_face.size();i++) {
            int x0 = _face[i-1].x;
            int x1 = _face[i].x;

            int y0 = _face[i-1].y;
            int y1 = _face[i].y;


            //largest values to appear as x1 or y1 are about 400,000
            //meshGridX -> CV_16S
            //meshGridY -> CV_16S
            //diffGPU -> CV_32S
            //xp1 and xp2 -> CV_32F
            //polyMaskGPU and binaryCompare -> CV_8U

            //get the x component of the vector formed between our line and vector formed between
            //the base of our line and every point in the mat
            cuda::subtract(meshGridX,x0,diffGPU);

            //multiply x component of every vector by y component of line
            cuda::multiply(diffGPU,y1 - y0,xp2);

            //get the y component of the vector formed between our line and vector formed between
            //the base of our line and every point in the mat
            cuda::subtract(meshGridY,y0,diffGPU);

            //multiply y component of every vector by x component of line
            cuda::multiply(diffGPU,x1 - x0,xp1);

            //subtraction as defined by cross product forumula
            cuda::subtract(xp2,xp1,xp1);

            //set all values less than 0 to 0, all values greater than 0 to 255
            cuda::compare(xp1,0,binaryCompare,CMP_GE);

            Mat temp;
            binaryCompare.download(temp);
            line(temp,_face[i-1],_face[i],Scalar(150),100);
            imwrite("/media/max/Data/binaryCompare.png",temp);

            cuda::multiply(polyMaskGPU,binaryCompare,polyMaskGPU);
        }
    }

    
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
            polyMaskGPU.upload(polyMaskOutput);

            update = true;
            images[i]->vertexId = res;

            //indicate that a new image has been added since last global alignment
            needsAlignment = true;

            //wait for buffer to be on gpu
            {
                std::unique_lock<std::mutex> lock(images[i]->cudaBufferMutex);
                images[i]->cudaBufferConVar.wait(lock, [&]{return images[i]->cudaBufferReady;});
            }

            //debayer image on gpu
            cuda::GpuMat image_Mat(image_size, CV_8U, images[i]->get_raw_cuda());
            cuda::cvtColor(image_Mat, threeChannelPrealGPU, COLOR_BayerBG2BGR);
            images[i]->free_memory_CUDA();

             if (componentMagLabel != 0) {
                 //flatfield correct
                 threeChannelPrealGPU.convertTo(convertHoldingGPU, CV_32F);
                 cuda::divide(convertHoldingGPU, ffGPU, convertHoldingGPU, 1, CV_32F);
                 //brighten
                 cuda::pow(convertHoldingGPU, 1.1, convertHoldingGPU);
                 convertHoldingGPU.convertTo(threeChannelPrealGPU, CV_8UC3);
             }

            //channelsGPU[0] = threeChannelPrealGPU; //3 channel
            cuda::split(threeChannelPrealGPU, channelsGPU);

            //add alpha channel
            channelsGPU.push_back(rectMaskGPU);
            cuda::merge(channelsGPU, fourChannelPrealGPU);


            //fourChannelPrealGPU.download(fourChannelPreallocated);


            //calculate effected tiles
            std::vector<Point2i> effectedTiles;
            std::vector<Point2i> effectedTilesNoMask;


            auto imageBox = cv::Rect_<float>(images[i]->absoluteCoords.x, images[i]->absoluteCoords.y, images[i]->width,
                                             images[i]->height);

            if (componentMagLabel == Image::_2X) {
                calculate_effected_tiles_round(face, effectedTiles, images[i]->absoluteCoords);
            } else {
                calculate_effected_tiles(face, effectedTiles, images[i]->absoluteCoords, &effectedTilesNoMask);
                imagePyramid->insertTilesAtBase(fourChannelPrealGPU, cuda::GpuMat(), imageBox, effectedTilesNoMask);
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
