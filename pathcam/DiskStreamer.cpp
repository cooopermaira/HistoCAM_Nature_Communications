//
//  DiskStreamer.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam{

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

}
