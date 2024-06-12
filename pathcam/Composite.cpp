//
//  Composite.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/24/23.
//
#define JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED 1

#include <stdio.h>
#include <pathCam.h>

namespace pathCam {

    CompositeVoronoi::CompositeVoronoi(StreamCam *parent, cv::Size image_size) : Composite(parent),
                                                                                 image_size(image_size) {
        subdiv_Bbox = Bbox(-50000, -50000, 50000, 50000);
        subdiv.initDelaunay(subdiv_Bbox.as_cvRect());
        circleMask = cv::Mat::zeros(image_size, CV_8U);
        cv::circle(circleMask, cv::Point(image_size.width / 2, image_size.height / 2), 2190, cv::Scalar(1), -1);
        std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(parent->imagePyramid);
        parent->imagePyramid->level.push_back(current);
        channels.resize(2);
        imageBoundsAsPolygon.resize(4);
        reset_image_as_polygon();


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

    void CompositeVoronoi::update(std::vector<RegInfo> new_info) {
        update_Bbox(new_info);
        expand_subdiv(new_info);
        add_images(new_info);
    }

    void CompositeVoronoi::add_images(std::vector<RegInfo> new_info) {

        // get a copy of references to all images at once so that only one mutex lock is needed
        std::vector<unsigned long int> indexes;
        for (int i = 0; i < new_info.size(); i++) {
            indexes.push_back(new_info[i].index);
        }
        std::vector<Image *> images = parent->get_image_refs(indexes);

        std::vector<iPoint> effectedTiles;
        for (int i = 0; i < images.size(); i++) {

            //calculate where the new image will be copied to in the composite
            Rect copyzone = Rect(new_info[i].absoluteCoords.x - root_offset.x,
                                 new_info[i].absoluteCoords.y - root_offset.y, images[i]->width, images[i]->height);

            //make copy of subdiv incase we decide not to use new point
            Subdiv2D tempSubdiv(subdiv);

            //add new point
            auto fShift = Point2f(new_info[i].absoluteCoords.x, new_info[i].absoluteCoords.y);
            int id = subdiv.insert(fShift);

            //get voronoi facets for only this face
            std::vector<std::vector<Point2f>> facets;
            std::vector<Point2f> centers;
            subdiv.getVoronoiFacetList({id}, facets, centers);

            //shift and recast
            std::vector<Point2i> face;
            for (auto &ii: facets[0]) {//we have pulled only one face so facets has only 1 element
                ii.x -= centers[0].x;
                ii.x += 6464 / 2;
                ii.y -= centers[0].y;
                ii.y += 4852 / 2;
                face.push_back((Point2i) ii);
            }

            //build polygon mask for new point
            Mat polyMaskOutput = cv::Mat::zeros(image_size, CV_8U);
            cv::fillConvexPoly(polyMaskOutput, face, cv::Scalar(255));

            //test for exclusion of frame via rollback
            int nonzeroMin;
            if (images[i]->label == Image::_2X) {
                polyMaskOutput = polyMaskOutput.mul(circleMask);
                nonzeroMin = 2190 * 2190 * 3.14 * 0.20;
            } else {
                nonzeroMin = images[i]->width * images[i]->height * 0.1;
            }

            if (countNonZero(polyMaskOutput) <= nonzeroMin) {
                //contributing less than x% of its pixels, revert and don't bother loading from disk
                subdiv = tempSubdiv;
                images[i]->free_memory_RAW();
                memberImages.push_back({images[i]->image_file.getFileName(), false});
                continue;
            }
            memberImages.push_back({images[i]->image_file.getFileName(), true});

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
            merge(channels,fourChannelPreallocated);

            fourChannelPreallocated.copyTo(composite(copyzone), polyMaskOutput);

            images[i]->free_memory_RAW();

            //calculate effected tiles
            calculate_effected_tiles(face, effectedTiles, new_info[i].absoluteCoords);
        }
        std::sort(effectedTiles.begin(), effectedTiles.end(), PointCompare<iPoint>());
        effectedTiles.erase(std::unique(effectedTiles.begin(), effectedTiles.end(), PointEquality<iPoint>()),
                            effectedTiles.end());


        tiledImageBounds = fRectangle((long) root_offset.x, (long) root_offset.y, composite.cols, composite.rows);
        parent->imagePyramid->level[0]->insertMatAtBase(composite, tiledImageBounds, effectedTiles);
        parent->imagePyramid->bounds = parent->imagePyramid->level[0]->bounds;

//      { const MessageManagerLock mmLock;
//        parent->parent->refreshImage();
//      }
      
      parent->update_observers();
/*
        parent->parent->imageview->setImage(parent->parent->MRimage);
        parent->parent->capture->setImage(parent->parent->MRimage);
        parent->parent->annotate->setImage(parent->parent->MRimage);
*/
    }


    void CompositeVoronoi::calculate_effected_tiles(std::vector<Point2i> maskAsPolygon, std::vector<iPoint> &result,
                                                    Vec2 absCoord) {
        std::vector<iPoint> tileIndices;
        std::map<int, std::vector<float>> tilesByColumn;

        //get the tile column of the left and right edges of the image frame
        int columnBoundLow = parent->imagePyramid->level[0]->getIJ(fPoint(absCoord.x, absCoord.y)).x;
        int columnBoundHigh = parent->imagePyramid->level[0]->getIJ(
                fPoint(absCoord.x + image_size.width, absCoord.y)).x;

        //get the tile row the top and bottom edges of the image frame
        long rowBoundLow = parent->imagePyramid->level[0]->getIJ(fPoint(absCoord.x, absCoord.y)).y;
        long rowBoundHigh = parent->imagePyramid->level[0]->getIJ(
                fPoint(absCoord.x, absCoord.y + image_size.height)).y;

        float yPixelBoundLow = float(rowBoundLow) * float(parent->imagePyramid->tile_size);
        float yPixelBoundHigh = float(rowBoundHigh) * float(parent->imagePyramid->tile_size);

        //for each edge of the voronoi mask
        for (int ii = 0; ii < maskAsPolygon.size(); ii++) {
            //if we are at the last point, make the next point the first point (this makes the last edge)
            int ii2 = (ii + 1) == maskAsPolygon.size() ? 0 : ii + 1;

            //create an fPoint from the cv::Point2i
            auto p1 = fPoint(maskAsPolygon[ii].x + absCoord.x, maskAsPolygon[ii].y + absCoord.y);
            auto p2 = fPoint(maskAsPolygon[ii2].x + absCoord.x, maskAsPolygon[ii2].y + absCoord.y);

            //test for duplicate points that result from the voronoi calculation
            if (p1.x == p2.x && p1.y == p2.y) {
                continue;
            }

            //retreive the tiles these points fall within
            auto tile1 = parent->imagePyramid->level[0]->getIJ(p1);
            auto tile2 = parent->imagePyramid->level[0]->getIJ(p2);

            //get the column of these tiles
            int column1 = tile1.getX();
            int column2 = tile2.getX();

            //determine which column is on the right and which is on the left
            int xlow = min(column1, column2);
            int xhigh = max(column1, column2);

            //take the column range bounds to be the most inward of the frame edges and the voronoi cell vertices
            //this accomplishes the same thing as taking the polygon intersection of the voronoi face and the image frame
            if( columnBoundHigh < xlow || columnBoundLow > xhigh){
                continue; //no intersection with frame tiles
            }
            xlow = max(columnBoundLow,xlow);
            xhigh = min(columnBoundHigh,xhigh);

            auto plow = p1.x < p2.x ? p1 : p2;
            auto phigh = p1.x >= p2.x ? p1 : p2;

            for (float j = xlow; j <= xhigh; j++) {

                float columnLeftEdge = j * float(parent->imagePyramid->level[0]->getTileSize());
                float columnRightEdge = (j + 1) * float(parent->imagePyramid->level[0]->getTileSize());
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

            if (lastPoint_y < yPixelBoundLow || firstPoint_y > yPixelBoundHigh){
                continue;
            }

            int lastTile = parent->imagePyramid->level[0]->getIJ(
                    fPoint(key * parent->imagePyramid->level[0]->getTileSize(), lastPoint_y)).getY();
            int firstTile = parent->imagePyramid->level[0]->getIJ(
                    fPoint(key * parent->imagePyramid->level[0]->getTileSize(), firstPoint_y)).getY();

            for (int ii = firstTile; ii <= lastTile; ii++) {
                if(ii <= rowBoundHigh && ii >= rowBoundLow) {
                    result.push_back(iPoint(key, ii));
                }
            }
        }
    }


    Composite::Composite(StreamCam *parent) : parent(parent), root_offset(0.0, 0.0), max_offset(0.0, 0.0) {
        flat_field = cv::imread(parent->flat_field_file.toString());
        flat_field.convertTo(flat_field, CV_32F);
        flat_field *= 1 / 170.0;
        //local_quality_score = score_image_2X(4852,6464,2190);
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
            Mat4b new_combined(int(max_offset.y - root_offset.y), int(max_offset.x - root_offset.x), Vec4b(0,0, 0, 0));
            //Mat new_combined_z_buffer = cv::Mat::zeros(cv::Size(new_combined.cols, new_combined.rows), CV_16U);

            if (composite.data) {

                Rect copyzone = Rect(temp_offset.x - root_offset.x, temp_offset.y - root_offset.y, composite.cols,
                                     composite.rows);

                composite.copyTo(new_combined(copyzone));
                //composite_z_buffer.copyTo(new_combined_z_buffer(copyzone));

            }

            composite = new_combined;
            //composite_z_buffer = new_combined_z_buffer;
            unsigned int topLogicSize = parent->imagePyramid->level.back()->getLogicSize();
            while (topLogicSize < composite.rows || topLogicSize < composite.cols ){
                unsigned int tile_size = parent->imagePyramid->level[0]->getTileSize();
                unsigned int logic_size = 2 * parent->imagePyramid->level.back()->getLogicSize();
                int levelWithinPyramid = parent->imagePyramid->level.size();
                std::shared_ptr<TiledImage> next_level = std::make_shared<TiledImage>(parent->imagePyramid,tile_size,logic_size,levelWithinPyramid);
                if(composite.data){
                    Mat temp;
                    resize(composite,temp,Size(composite.cols / pow(2, levelWithinPyramid), composite.rows / pow(2, levelWithinPyramid)));
                    tiledImageBounds = fRectangle((long) root_offset.x, (long) root_offset.y, composite.cols, composite.rows);
                    next_level->insertMat(temp,tiledImageBounds);
                }
                parent->imagePyramid->level.push_back(next_level);
                topLogicSize = logic_size;
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


    long CompositeVoronoi::segment_yval_at_point(float xloc, fPoint p1, fPoint p2) {
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

        circle(mask, cv::Point(img.cols / 2, img.rows / 2), 2190, cv::Scalar(1), -1);
        cv::Mat temp = mask.mul(img);
        imwrite("mask.png", temp);
        //imwrite("img.png", img);
        return temp;
    }

    void CompositeVoronoi::debug_write_contribution_on_grid(std::string name ,pathCam::Vec2 absCoord, cv::Mat &img, cv::Mat &mask) {
        int k = int(absCoord.x);
        k = 512 - k%512;
        if(k < 0){k=abs(k);}
        Mat debugmat = Mat::zeros(4852,6464,CV_8UC4);
        img.copyTo(img,mask);
        for (int m = k; m < debugmat.cols; m += 512){
            cv::line(debugmat,cv::Point(m,0),cv::Point(m,4852),Scalar(0,0,255,255));
            cv::line(debugmat,cv::Point(m+1,0),cv::Point(m+1,4852),Scalar(0,0,255,255));
            cv::line(debugmat,cv::Point(m+2,0),cv::Point(m+2,4852),Scalar(0,0,255,255));

        }
        k = int(absCoord.y);
        k = 512 - k%512;

        if(k < 0){k=abs(k);}
        for (int m = k; m < debugmat.rows; m += 512){
            cv::line(debugmat,cv::Point(0,m),cv::Point(6464,m),Scalar(0,0,255,255));
            cv::line(debugmat,cv::Point(0,m+1),cv::Point(6464,m+1),Scalar(0,0,255,255));
            cv::line(debugmat,cv::Point(0,m+2),cv::Point(6464,m+2),Scalar(0,0,255,255));

        }
        imwrite(name,debugmat);
    }
}

