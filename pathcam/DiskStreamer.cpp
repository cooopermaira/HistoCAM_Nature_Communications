//
//  DiskStreamer.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include <utility>

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"

using Poco::DirectoryIterator;

namespace pathCam {
    DiskReader::DiskReader(StreamCam *parent): parent(parent), successful(false) {
        parent->microscopeInput = true;
    }

    void DiskReader::run() {
        std::ifstream infile(parent->input_images.toString().c_str());
        std::string imageFile;
        unsigned long sort_order = 0;
        while (infile >> imageFile) {
            /*
            auto ds = new DiskStreamer(parent, imageFile, sort_order);
            parent->diskCount++;
            parent->JobQ->add_runnable(ds);
            sort_order += 10;
             */
            Image* image = new Image();
            image->set_disk_file(imageFile);
            parent->pass_image(image, sort_order);
            sort_order++;
        }
        parent->microscopeInput = false;
        std::cout << "disk images set " << std::endl;
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

    void DebayerRunnable::run() {
        image->load_raw_from_disk();

        if(!image->in_memory() ){
            std::cout << "Issue loading image.\n";
            return;
        }

        //image->create_reg_image(1.0,1.0,true,cv::INTER_CUBIC, false);
        cv::Size image_size(image->width, image->height);
        Mat image_Mat = cv::Mat(image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
        cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);

        Path o = outfile;
        o.append(image->image_file.getFileName());
        o.setExtension("png");

        imwrite(o.toString(), image_Mat);
    }
}
