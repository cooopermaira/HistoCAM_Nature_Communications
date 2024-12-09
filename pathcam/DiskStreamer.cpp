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


  DiskStreamer::DiskStreamer(StreamCam *parent, std::string imageFile,
                             unsigned long sort_order) : parent(parent),
                                                         imageFile(std::move(imageFile)),
                                                         RunnableIntermediate(sort_order, 0) {
  };

  void DiskStreamer::run() {
    if (imageFile.empty()) { return; }
    std::ifstream stream;
    stream.open(imageFile, std::ios::binary);
    unsigned int height = parent->image_height;
    unsigned int width = parent->image_width;
    char *raw_image_data = new char[width * height];
    stream.read(raw_image_data, width * height);

    Image *image = new Image(width, height, parent->scope_radius);
    image->copy_in(raw_image_data);
    image->set_disk_file(imageFile);
    delete[] raw_image_data;

    parent->pass_image(image, sort_order + 1);

    parent->diskCount--;
    successful = true;
  }

  void DebayerRunnable::run() {
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

      cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);

      cv::divide(image_Mat, flatfield, image_Mat, 1.0, CV_8U);

      Poco::Path o = outfile;
      o.append(image->image_file.getFileName());
      o.setExtension("png");

      imwrite(o.toString(),image_Mat);
    }
    image->free_memory_RAW();
    int k = 0;
  }
}
