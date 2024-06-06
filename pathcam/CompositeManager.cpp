//
//  CompositeManager.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//
// This class listens for registered frames, sorts them by component membership, then passes them to the component object for compositing.
#define JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED 1

#include "pathCam.h"

//class MRTiledImage;
//class TiledImage;

namespace pathCam {

    CompositeManager::CompositeManager(StreamCam *parent) : parent(parent), successful(false) {};

    void CompositeManager::run() {

        while (parent->microscopeInput || parent->diskCount > 0 || parent->regCount > 0 || parent->loaderCount > 0 ||
               parent->matchableCount > 0 || !parent->compositeQ_empty()) {

            //if nothing in the Q but termination condition not met, wait
            if (parent->compositeQ_empty()) {
                Poco::Thread::sleep(100);

            } else {

                std::vector<RegInfo> indexes = parent->get_Q_front();
                std::sort(indexes.begin(), indexes.end());

                while (parent->composites.size() <= indexes.back().component_membership) {
                    //Because of the multithreading, this place in the code can be reached before a new component object has been instantiated and added to the vector. If this happens, wait.
                    Poco::Thread::sleep(100);
                }

                unsigned int current_component = indexes[0].component_membership;
                std::vector<RegInfo> new_info;

                //sort the new frames by component and pass them to their respective components for compositing.
                for (int i = 0; i < indexes.size(); i++) {

                    if (current_component == indexes[i].component_membership) {
                        new_info.push_back(indexes[i]);
                    } else {
                        parent->composites[current_component]->update(new_info);
                        current_component = indexes[i].component_membership;
                        new_info.clear();
                    }
                }
                parent->composites[current_component]->update(new_info);

            }
        }

        if (parent->withFrontEnd) {

            Mat image_in = parent->composites[0]->get_composite();

            unsigned int height = image_in.rows;
            unsigned int width = image_in.cols;

            //parent->imagePyramid->bounds = fRectangle(0, 0, width, height);

            unsigned int tile_size = parent->imagePyramid->tile_size;
/*
            double total_levels = ceil(max(log2(width), log2(height)) - log2(tile_size) + 1);
            double total_pixels = 0.0;
            for (unsigned int i = 0; i < total_levels; i++) {
                total_pixels += (width / (pow(2, i))) * (height / (pow(2, i)));
            }


            std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(parent->imagePyramid,tile_size, tile_size,0);
            current->insertMat(image_in, fRectangle(0, 0, width, height));

            parent->imagePyramid->level.push_back(current);
*/
/*
            unsigned int num_levels = 1;
            while (image_in.cols > tile_size || image_in.rows > tile_size) {
                std::shared_ptr<TiledImage> current = std::make_shared<TiledImage>(parent->imagePyramid,tile_size,tile_size * pow(2, num_levels),num_levels);
                //cv::resize causes a reallocation and is possibly better done with cv::pyrDown()
                cv::resize(image_in, image_in, cv::Size(image_in.cols / 2, image_in.rows / 2));

                //current->insertMat(image_in, fRectangle(0, 0, width, height));
                current->insertMat(image_in,parent->imagePyramid->bounds);
                parent->imagePyramid->level[num_levels]=current;
                num_levels += 1;

            }
            */

            for (int i = 1; i < parent->imagePyramid->level.size(); i++){
                cv::resize(image_in, image_in, cv::Size(image_in.cols / 2, image_in.rows / 2));
                parent->imagePyramid->level[i]->insertMat(image_in,parent->imagePyramid->bounds);
            }
        }


        /*
        for (int i = 0; i < parent->composites.size(); i++){
          std::cout << "Writing image of size: " << parent->composites[i]->get_composite().size() << "\n";
          imwrite("finish" + std::to_string(i) + ".png", parent->composites[i]->get_composite());
        }
      */
    }

}
