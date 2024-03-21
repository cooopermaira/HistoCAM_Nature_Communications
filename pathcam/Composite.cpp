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
  
  //root_offset is the distance from (0,0) of the cv image to the root frame, which is (0,0) in registration space. max_offset is the distance from (0,0) in registration space to the bottom right corner of the cv image. Total dimensions of image are max_offset - root_offset.
  Vec2 temp_offset = root_offset;
  
  // If any new frames extend beyond the current extent, expand cv image dimensions
  for(int i = 0; i < new_info.size(); i++){
    if(new_info[i].vec.x < root_offset.x){
      update_box = true;
      root_offset.x = new_info[i].vec.x;
    }
    if(new_info[i].vec.y < root_offset.y){
      update_box = true;
      root_offset.y = new_info[i].vec.y;
    }
    
    if(new_info[i].vec.x + 6464 > max_offset.x){
      update_box = true;
      max_offset.x = new_info[i].vec.x + 6464;
    }
    if(new_info[i].vec.y + 4852 > max_offset.y){
      update_box = true;
      max_offset.y = new_info[i].vec.y + 4852;
    }
  }
  
  if (update_box){
    //if we are updating the bounding box, create a new combined image and copy old image into the correct location
    Mat3b new_combined(int(max_offset.y - root_offset.y),int(max_offset.x - root_offset.x), Vec3b(0,0,0));
    Mat new_combined_z_buffer = cv::Mat::zeros(cv::Size(new_combined.cols, new_combined.rows), CV_8U);
    
    if (composite.data){
      
      Rect copyzone = Rect(temp_offset.x - root_offset.x, temp_offset.y - root_offset.y, composite.cols, composite.rows);
      
      composite.copyTo(new_combined(copyzone));
      composite_z_buffer.copyTo(new_combined_z_buffer(copyzone));
      
    }
    
    composite = new_combined;
    composite_z_buffer = new_combined_z_buffer;
    
  }
};




void Composite::add_images(std::vector < RegInfo > new_info){
  
  std::vector<unsigned long int> indexes;
  
  for (int i = 0; i < new_info.size(); i++){
    indexes.push_back(new_info[i].index);
  }
  
  // get a copy of references to all images at once so that only one mutex lock is needed
  std::vector<Image*> images = parent->get_image_refs(indexes);
  
  //this may need to be placed inside the below for loop if frames ever vary in size. For now it is here so the mask only needs to be built once
  cv::Size image_size(images[0]->width,images[0]->height);
  
  Mat mask = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);
  
  circle(mask, cv::Point(image_size.width/2, image_size.height/2), 2190, cv::Scalar(255), -1);
  

  for (int i = 0; i < images.size(); i++){
    //calculate where the new image will be copied to in the composite
    Rect copyzone = Rect(new_info[i].vec.x - root_offset.x, new_info[i].vec.y - root_offset.y,                                                     images[i]->width, images[i]->height);
    
    //calculate which pixels of the new image will be copied into the composite
    Mat use_locations = mask.mul(local_quality_score > composite_z_buffer(copyzone));
    
    
    if (countNonZero(use_locations) == 0){
      continue; //not contributing, don't bother loading from disk
    }
    
    images[i]->load_raw_from_disk();
    Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
    cvtColor(image_Mat,image_Mat,COLOR_BayerBG2BGR);
    cv::divide(image_Mat,flat_field,image_Mat,1.0,CV_8U);
    
    image_Mat.copyTo(composite(copyzone),use_locations);
    local_quality_score.copyTo(composite_z_buffer(copyzone),use_locations);
    
    images[i]->free_memory_RAW();
  }
  /*
  imshow("display",composite);
  waitKey(10);
  */
}

void Composite::update(std::vector < RegInfo > new_info){
  update_Bbox(new_info);
  add_images(new_info);
}

Mat Composite::get_composite(){
  return composite;
}
}

