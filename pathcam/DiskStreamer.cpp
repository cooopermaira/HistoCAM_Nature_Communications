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
    Mat ConvertBGR2Bayer(Mat BGRImage) {
        /*
        Assuming a Bayer filter that looks like this:

        # // 0  1  2  3  4  5
        /////////////////////
        0 // B  G  B  G  B  G
        1 // G  R  G  R  G  R
        2 // B  G  B  G  B  G
        3 // G  R  G  R  G  R
        4 // B  G  B  G  B  G
        5 // G  R  G  R  G  R

        */


        Mat BayerImage(BGRImage.rows, BGRImage.cols, CV_8UC1);

        int channel;

        for (int row = 0; row < BayerImage.rows; row++) {
            for (int col = 0; col < BayerImage.cols; col++) {
                if (row % 2 == 0) {
                    //even columns and even rows = blue = channel:0
                    //even columns and uneven rows = green = channel:1
                    channel = (col % 2 == 0) ? 0 : 1;
                } else {
                    //uneven columns and even rows = green = channel:1
                    //uneven columns and uneven rows = red = channel:2
                    channel = (col % 2 == 0) ? 1 : 2;
                }

                BayerImage.at<uchar>(row, col) = BGRImage.at<Vec3b>(row, col).val[channel];
            }
        }

        return BayerImage;
    }


    DiskReader::DiskReader(StreamCam *parent) : parent(parent) {
        parent->microscopeInput = true;
    }

    void DiskReader::run() {
        std::ifstream infile(parent->input_images.toString().c_str());
        std::string imageFile;
        unsigned long image_index = 0;
        while (infile >> imageFile) {
            Image *image = new Image(parent->image_width, parent->image_height, parent->scope_radius);
            image->set_disk_file(imageFile);
            parent->pass_image(image, image_index);
            image_index++;
        }
        parent->microscopeInput = false;
        std::cout << "disk images set " << std::endl;
    }


    void DebayerRunnable::run() {
        Poco::Path o = outfile;

        image->load_raw_from_disk();

        bool convertAndSave = true;

        if (!image->in_memory()) {
            std::cout << "Issue loading image.\n";
            return;
        }
        auto val = image->check_blur();
        blur->at(sort_order) = val;
        names->at(sort_order) = image->get_ImageFile().getFileName();

        if (convertAndSave) {
            //image->create_reg_image(1.0,1.0,true,cv::INTER_CUBIC, false);
            cv::Size image_size(image->width, image->height);
            Mat image_Mat = cv::Mat(image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
            Mat imwriteMat;

            cvtColor(image_Mat, image_Mat, COLOR_BayerBG2RGB);
            try {
                cv::divide(image_Mat, flatfield, image_Mat, 1.0, CV_8U);

                image_Mat.convertTo(image_Mat, CV_32FC3);

                cv::pow(image_Mat, 1.08, image_Mat);

                image_Mat.convertTo(image_Mat, CV_8UC3);

                //add subdir for png
                auto r = o;
                // o.pushDirectory("png");
                // o.setFileName(image->get_ImageFile().getFileName());
                // o.setExtension("png");
                // imwrite(o.toString(), image_Mat);
                //
                // cvtColor(image_Mat,image_Mat, COLOR_BayerBG2RGB);
                image_Mat = ConvertBGR2Bayer(image_Mat);

                //save .Raw
                r.setFileName(image->get_ImageFile().getFileName());
                r.setExtension("Raw");
                std::fstream file;
                file = std::fstream(r.toString(), std::ios::out | std::ios::binary);
                if (file.fail()) {
                    throw new std::exception;
                }
                file.write(reinterpret_cast<const char *>(image_Mat.data), image->width * image->height);
            } catch (...) {
                int k = 0;
            }
        }
        image->free_memory_RAW();
        int k = 0;
    }
}
