//
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"


using namespace pathCam;



int main(int argc, char** argv){
  
  std::string test_file = "/Users/bsumma/Source/tulane/pathcam/resources/frame-10282021160735-3.tiff";
  std::string flat_file = "/Users/bsumma/Source/tulane/pathcam/resources/2x_wb.tiff";
  
  cv::Mat flat_field = cv::imread(flat_file.c_str());
  std::cout << flat_field.rows << " x " << flat_field.cols << "\n";
  
  int rows = flat_field.rows;
  int cols = flat_field.cols;
  
  cv::Mat image = cv::imread(test_file.c_str());
  std::cout << image.rows << " x " << image.cols << "\n";
  
  flat_field.convertTo(flat_field, CV_32F);
  flat_field *= 1/170.0;

  image.convertTo(image, CV_32F);
  cv::divide(image, flat_field, image, 1.0, CV_32F);
  
  // Convert the corrected image back to 8-bit format for displaying
  image.convertTo(image, CV_8U);

  Mat mask = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);
  circle(mask, cv::Point(image.cols/2, image.rows/2), 2190, cv::Scalar(255, 255, 255), -1);

  cv::bitwise_and(image, mask, image);
  
  imwrite("/Users/bsumma/Source/tulane/pathcam/resources/output.tiff", image);
  
  return 0;
}
