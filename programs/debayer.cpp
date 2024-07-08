//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"

using Poco::DirectoryIterator;

int main(int argc, char *argv[]) {

    //Should probably add fancier command line parsing
    if (argc < 3) {
        std::cout << "Missing input. Use:\n";
        std::cout << "debayer <path to input image> <path to output image>\n";
        return -1;
    }

    auto inFile = Poco::Path(argv[1]);
    auto outFile = Poco::Path(argv[2]);
    //Add in argument to specify size of output image in format widthXheight default value is 6464x4852
    unsigned int width;
    unsigned int height;

    if (argc == 4) {
        std::string sizeArg = argv[3];
        size_t xPos = sizeArg.find('x');
        if (xPos != std::string::npos) {
            std::string widthStr = sizeArg.substr(0, xPos);
            std::string heightStr = sizeArg.substr(xPos + 1);
            try {
                width = std::stoul(widthStr);
                height = std::stoul(heightStr);
            }
            catch (const std::invalid_argument& e) {
                std::cout << "Invalid size format. Use widthXheight.\n";
                return -1;
            }
            catch (const std::out_of_range& e) {
                std::cout << "Size value out of range.\n";
                return -1;
            }
        }
        else {
            std::cout << "Invalid size format. Use widthXheight.\n";
            return -1;
        }
    }
    else {
        width = 6464;
        height = 4852;
    }

    if (!(inFile.isDirectory() == outFile.isDirectory())) {
        std::cout << "Input needs to be both directories or files.\n";
        return -1;
    }

    if (inFile.isDirectory()) {

        std::cout << "Processing Directories\n";
        auto jq = pathCam::JobQueue(20, 20);

        Poco::DirectoryIterator it(inFile);
        Poco::DirectoryIterator end;
        while (it != end) {

            Poco::Path p(it.path());

            if (p.getExtension() == "Raw") {
                //std::cout << "read:" << p.toString() << "\n";
                
                auto *image = new pathCam::Image(width, height);

                image->set_disk_file(p);
                auto *dr = new pathCam::DebayerRunnable(image, outFile);
                jq.add_runnable(dr);

            }
            ++it;
        }
        while (!jq.is_empty()) {
          jq.run_jobs(false);
        }


    } else {

        std::cout << "Processing File\n";

        pathCam::Image* image = new pathCam::Image(width, height);

        image->set_disk_file(inFile);
        image->load_raw_from_disk();


        if (!image->in_memory()) {
            std::cout << "Issue loading image.\n";
            return -1;
        }

        image->create_reg_image(1.0, 1.0, true, cv::INTER_CUBIC, false);

        imwrite(outFile.toString(), image->get_reg_image());

        delete image;
    }

    return 0;

}
