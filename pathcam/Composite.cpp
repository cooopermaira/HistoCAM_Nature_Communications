//
//  Composite.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/24/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam{

Composite::Composite(StreamCam *parent): parent(parent), root_offset(0.0,0.0),max_offset(0.0, 0.0){
  flat_field = cv::imread(parent->flat_field_file.toString());
  flat_field.convertTo(flat_field, CV_32F);
  flat_field *= 1/170.0;
  local_quality_score = score_image_2X(4852,6464,2190);
}

Mat Composite::score_image_2X(int rows, int cols, int radius){
  
  Mat img = cv::Mat_<uint8_t>(rows,cols);
  double idist,jdist,rad_sq;
  rad_sq = pow(radius,2);
  
  for (int i = 0;i < img.rows;i++){
    idist = pow(i - img.rows/2,2);
    
    for (int j = 0; j < img.cols;j++){
      jdist = pow(j - img.cols/2,2);
      img.at<uint8_t>(i,j) = std::max( (uint8_t) 0 , uint8_t( floor( 255.0 / rad_sq * (rad_sq - idist - jdist) ) ) );
    }
  }
  return img;
}



void Composite::update_Bbox(std::vector < RegInfo > new_info){
  bool update_box = false;
  Vec2 temp_offset(0.0,0.0);
  for(int i = 0; i < new_info.size(); i++){
    if(new_info[i].vec.x < root_offset.x){
      update_box = true;
      temp_offset.x = root_offset.x - new_info[i].vec.x;
      root_offset.x = new_info[i].vec.x;
    }
    if(new_info[i].vec.y < root_offset.y){
      update_box = true;
      temp_offset.y = root_offset.y - new_info[i].vec.y;
      root_offset.y = new_info[i].vec.y;
    }
    
    if(new_info[i].vec.x + 6464 > max_offset.x){
      update_box = true;
      max_offset.x = new_info[i].vec.x;
    }
    if(new_info[i].vec.y + 4852 > max_offset.y){
      update_box = true;
      max_offset.y = new_info[i].vec.y;
    }
  }
  
  if (update_box){
    
    Mat3b new_combined(int(max_offset.y - root_offset.y),int(max_offset.x - root_offset.x), Vec3b(0,0,0));
    Mat3f new_flatfield_buffer(new_combined.rows,new_combined.cols);
    Mat new_combined_z_buffer = cv::Mat::zeros(cv::Size(new_combined.cols, new_combined.rows), CV_8U);
    
    Rect copyzone = Rect(temp_offset.y,temp_offset.x,composite.cols,composite.rows);
    
    composite.copyTo(new_combined(copyzone));
    flat_field_composite.copyTo(new_flatfield_buffer(copyzone));
    composite_z_buffer.copyTo(new_combined_z_buffer(copyzone));
    
    composite = new_combined;
    flat_field_composite = new_flatfield_buffer;
    composite_z_buffer = new_combined_z_buffer;
  };
}



void Composite::add_images(std::vector < RegInfo > new_info){
  
  std::vector<unsigned long int> indexes;
  
  for (int i = 0; i < new_info.size(); i++){
    indexes.push_back(new_info[i].index);
  }
  
  std::vector<Image*> images = parent->get_image_refs(indexes);
  cv::Size image_size(images[0]->width,images[0]->height);
  
  Mat mask = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);
  
  circle(mask, cv::Point(image_size.width/2, image_size.height/2), 2190, cv::Scalar(255), -1);
  
  for (int i = 0; i < images.size(); i++){
    Rect copyzone = Rect(root_offset.x + new_info[i].vec.x, root_offset.y + new_info[i].vec.y,                                                     images[i]->width, images[i]->height);
    
    Mat use_locations = mask.mul(local_quality_score > composite_z_buffer(copyzone));
    
    if (countNonZero(use_locations) == 0){continue;}//not contributing
    
    images[i]->load_raw_from_disk();
    
    Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
    cvtColor(image_Mat,image_Mat,COLOR_BayerBG2BGR);
    
    image_Mat.copyTo(composite(copyzone),use_locations);
    flat_field.copyTo(flat_field_composite(copyzone),use_locations);
    local_quality_score.copyTo(composite_z_buffer(copyzone),use_locations);
  }
  cv::divide(composite,flat_field_composite,composite,1.0,CV_8U);
}

void Composite::update(std::vector < RegInfo > new_info){
  update_Bbox(new_info);
  add_images(new_info);
}

Mat Composite::get_composite(){
  return composite;
}
}

