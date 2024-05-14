//
//  DiskStreamer.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <utility>

#include "pathCam.h"

namespace pathCam {
    DiskReader::DiskReader(StreamCam *parent): parent(parent), successful(false) {
        parent->microscope_input = true;
    }

    void DiskReader::run() {
        std::ifstream infile(parent->input_images.toString().c_str());
        std::string imageFile;
        unsigned long sort_order = 0;
        while (infile >> imageFile) {
            auto ds = new DiskStreamer(parent, imageFile, sort_order);
            parent->diskCount++;
            parent->JobQ->add_runnable(ds);
            sort_order += 10;
        }
        parent->microscope_input = false;
        parent->microscope_input = false;
        std::cout << "disk images loaded " << std::endl;
    }


    DiskStreamer::DiskStreamer(StreamCam *parent, std::string imageFile,
                               unsigned long sort_order): parent(parent),
                                                          imageFile(std::move(imageFile)),
                                                          successful(false), RunnableIntermediate(sort_order){
    };

    void DiskStreamer::run() {
        if (imageFile.empty()) { return; }
        std::ifstream stream;
        stream.open(imageFile, std::ios::binary);

        char *raw_image_data = new char[6464 * 4852];
        stream.read(raw_image_data, 6464 * 4852);

        Image *image = new Image();
        image->copy_in(raw_image_data);
        image->set_disk_file(imageFile);
        delete [] raw_image_data;

        parent->pass_image(image, sort_order + 1);
        parent->diskCount--;
        successful = true;
    }

    /*
        DiskStreamer::DiskStreamer(StreamCam *parent): parent(parent), successful(false){};

        void DiskStreamer::run() {
            parent->microscope_input = true;

            std::ifstream infile(parent->input_images.toString().c_str());
            std::string imageFile;
            while (infile >> imageFile) {
                if (imageFile.size() == 0) { continue; }
                std::ifstream stream;
                stream.open(imageFile, std::ios::binary);

                char* raw_image_data = new char[6464 * 4852];
                stream.read(raw_image_data, 6464 * 4852);

                Image* image = new Image();
                image->copy_in(raw_image_data);
                image->set_disk_file(imageFile);
                delete [] raw_image_data;

                parent->pass_image(image);
            }
            parent->microscope_input = false;
            successful = true;
            std::cout << "disk empty ";
        }

    */
}
