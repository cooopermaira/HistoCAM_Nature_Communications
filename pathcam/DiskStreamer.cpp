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
    image->create_reg_image(.25,.5,true,INTER_CUBIC,false);
    Mat temp = image->get_reg_image();
    image->sharpness = image->compute_sharpness(temp);
    // Size image_size(image->width, image->height);
    // Mat image_Mat = Mat(image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
    // cvtColor(image_Mat, image_Mat, COLOR_BayerBG2GRAY);
    // Point2f ans;
    //
    // Rect roi(image_Mat.cols/2-64,image_Mat.rows/2-64,128,128);
    // Mat temp = image_Mat(roi);
    // ans = image->compute_sharpness(temp);
    // imwrite("/media/max/Data/blur_test/roi.png",temp);
    // for (int i = 0; i < 5; ++i) {
    //   resize(image_Mat,image_Mat,Size(image_Mat.cols/2,image_Mat.rows/2),0,0,INTER_CUBIC);
    //   roi = Rect(image_Mat.cols/2-64,image_Mat.rows/2-64,128,128);
    //   Mat temp = image_Mat(roi);
    //   ans = image->compute_sharpness(temp);
    //   imwrite("/media/max/Data/blur_test/roi.png",temp);
    //   int k = 0;
    // }

    return;

    bool convertAndSave = true;

    if (!image->in_memory()) {
      std::cout << "Issue loading image.\n";
      return;
    }
    //    auto val = image->check_blur();
    //        blur->at(sort_order) = val;
    //        names->at(sort_order) = image->get_ImageFile().getFileName();

    if (convertAndSave) {

      std::vector<std::string> ffs={"/media/max/Data/2_20/2x/cal/2x_cal.Raw","/media/max/Data/2_20/4x/cal/4x_cal.Raw",
            "/media/max/Data/2_20/10x/cal/10x_cal.Raw",
            "/media/max/Data/2_20/20x/cal/20x_cal.Raw"};

      Size image_size(image->width, image->height);
      Mat image_Mat;
      Mat readMat = Mat(image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
      cvtColor(readMat, readMat, COLOR_BayerBG2BGR);

      resize(readMat,readMat,Size(image->width / 4, image->height / 4));
      auto r = outfile;

      r.setFileName(image->get_ImageFile().getBaseName());
      r.setExtension("png");
      imwrite(r.toString(), readMat);
    //   readMat.convertTo(readMat,CV_32FC3);
    //
    //   for (int i = 0; i < ffs.size(); ++i) {
    //     Mat ff;
    //     std::ifstream stream;
    //     stream.open(ffs[i], std::ios::binary);
    //     {
    //       char *raw_buffer = new char[6464 * 4852];
    //       stream.read(raw_buffer, 6464 * 4852);
    //       stream.close();
    //       ff = Mat(Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);
    //     }
    //     Mat gray;
    //     cvtColor(ff,gray,COLOR_BayerBG2GRAY);
    //     resize(gray,gray,Size(image->width / 4, image->height / 4));
    //     cvtColor(ff, ff, COLOR_BayerBG2BGR);
    //     imwrite("/media/max/Data/2_20/wrong_flatfield_test/png2/ff_"+std::to_string(i)+".png",gray);
    //     ff.convertTo(ff, CV_32F);
    //     ff *= 1 / 170.0;
    //     divide(readMat, ff, image_Mat, 1, CV_32F);
    //
    //     //cv::pow(image_Mat, 1.1, image_Mat);
    //
    //     image_Mat.convertTo(image_Mat, CV_8UC3);
    //
    //
    //     resize(image_Mat, image_Mat, Size(image->width / 4, image->height / 4));
    //     //
    //     //      imwrite("/Users/coopermaira/Desktop/ff.png", flatfield);
    //     //      imwrite("/Users/coopermaira/Desktop/pre_ff.png",image_Mat);
    //
    //
    //     //divide(readMat, flatfield, image_Mat, 1, CV_32F);
    //
    //     //cv::pow(image_Mat, 1.1, image_Mat);
    //
    //     //image_Mat.convertTo(image_Mat, CV_8UC3);
    //
    //     //add subdir for png
    //     auto r = outfile;
    //
    //     r.setFileName(image->get_ImageFile().getBaseName()+"_"+std::to_string(i));
    //     r.setExtension("png");
    //     imwrite(r.toString(), image_Mat);
    //   }
    //   //
    //   // cvtColor(image_Mat,image_Mat, COLOR_BayerBG2RGB);
    //   //        image_Mat = ConvertBGR2Bayer(image_Mat);
    //   //
    //   //        //save .Raw
    //   //        auto name = std::stoi(image->get_ImageFile().getBaseName());
    //   //        name += 250;
    //   //
    //   //
    //   //        r.setFileName(std::to_string(name));
    //   //        r.setExtension("Raw");
    //   //        std::fstream file;
    //   //        file = std::fstream(r.toString(), std::ios::out | std::ios::binary);
    //   //        if (file.fail()) {
    //   //          throw new std::exception;
    //   //        }
    //   //        file.write(reinterpret_cast<const char *>(image_Mat.data), image->width * image->height);
    //
    //   // } catch (cv::Exception &e) {
    //   //   int k = 0;
    //   // }
     }
    image->free_memory_RAW();
    int k = 0;
  }
}
