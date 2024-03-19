//
//  Composite.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/24/23.
//

#include <stdio.h>
#include <pathCam.h>

namespace pathCam{

CompositeVoronoi::CompositeVoronoi(StreamCam* parent) : Composite(parent) {        
    subdiv_Bbox = Bbox(-5000000, -5000000, 5000000, 5000000);
    subdiv.initDelaunay(subdiv_Bbox.as_cvRect());

}

void CompositeVoronoi::expand_subdiv(std::vector < RegInfo > new_info) {
    bool extend = false;
    if (root_offset.x < subdiv_Bbox.min_x) {
        extend = true;
        subdiv_Bbox.min_x *= 2;
    }
    if (root_offset.y < subdiv_Bbox.min_y) {
        extend = true;
        subdiv_Bbox.min_y *= 2;
    }
    if (max_offset.x > subdiv_Bbox.max_x) {
        extend = true;
        subdiv_Bbox.max_x *= 2;
    }
    if (max_offset.y > subdiv_Bbox.max_y) {
        extend = true;
        subdiv_Bbox.max_y *= 2;
    }
    if (extend) {
        subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
        subdiv = Subdiv2D(subdiv_Bbox.as_cvRect());
        subdiv.insert(centers);
    }
}

void CompositeVoronoi::update(std::vector < RegInfo > new_info) {
    update_Bbox(new_info);
    expand_subdiv(new_info);
    add_images(new_info);
}

void CompositeVoronoi::add_images(std::vector < RegInfo > new_info) {

    cv::Size image_size(6464, 4852);
    for (int i = 0; i < new_info.size(); i++) {
        Subdiv2D temp(subdiv);
        subdiv.insert(cv::Point2f(new_info[i].vec.x, new_info[i].vec.y));
    }
    //Mat img = Mat::zeros(Size(3000, 3000), CV_8U);
    //draw_delaunay(img, subdiv, cv::Scalar(0, 0, 255));
    //imwrite("delauney.png", img);
    std::vector<std::vector<Point2f>> facets;
    std::vector<Point2f> centers;
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);
    for (int i = 0; i < new_info.size(); i++) {
        std::vector<Point2i> oneFacet;
        for (int j = 0; j < facets[i].size();j++) {
            facets[i][j].x -= centers[i].x ;
            facets[i][j].x += 6464 / 2;
            facets[i][j].y -= centers[i].y;
            facets[i][j].y += 4852 / 2;
            oneFacet.push_back((Point2i)facets[i][j]);
        }
        Mat mask = cv::Mat::zeros(image_size, CV_8U);
        cv::fillConvexPoly(mask, oneFacet,cv::Scalar(255));
        imwrite(std::to_string(i) + ".png", mask);
    }
    std::cout << "done";
}

Composite::Composite(StreamCam *parent): parent(parent), root_offset(0.0,0.0),max_offset(0.0, 0.0){
  flat_field = cv::imread(parent->flat_field_file.toString());
  flat_field.convertTo(flat_field, CV_32F);
  flat_field *= 1/170.0;
  local_quality_score = score_image_2X(4852,6464,2190);
}

cv::Mat Composite::score_image_2X(int rows, int cols, int radius){

  cv::Mat img = cv::Mat::zeros(cv::Size(cols, rows), CV_16U);
  //cv::Mat img = cv::Mat_<uint16_t>(rows,cols);
  double idist,jdist,rad_sq;
  rad_sq = pow(radius,2);
  double k = 0;
  for (int i = 0;i < radius;i++){
    
    
    for (int j = 0; j < radius;j++){
        uint16_t x = rows / 2 - radius + i+1;
        uint16_t y = cols / 2 - radius + j+1;
        /*
        if (j < i) {
            img.at<uint16_t>(x,y) = j;
            img.at<uint16_t>(img.rows - x, y) = j;
            img.at<uint16_t>(x,img.cols - y) = j;
            img.at<uint16_t>(img.rows -x, img.cols - y) = j;
        }
        else {
            img.at<uint16_t>(x, y) = i;
            img.at<uint16_t>(img.rows - x, y) = i;
            img.at<uint16_t>(x, img.cols - y) = i;
            img.at<uint16_t>(img.rows - x, img.cols - y) = i;
        }
        */
        img.at<uint16_t>(x, y) = radius - sqrt(pow(x - img.rows / 2, 2) + pow(y - img.cols, 2));
        img.at<uint16_t>(img.rows - x, y) = radius - sqrt(pow(x - img.rows / 2, 2) + pow(y - img.cols, 2));
        img.at<uint16_t>(x, img.cols - y) = radius - sqrt(pow(x - img.rows / 2, 2) + pow(y - img.cols, 2));
        img.at<uint16_t>(img.rows - x, img.cols - y) = radius - sqrt(pow(x - img.rows / 2, 2) + pow(y - img.cols, 2));
    }
    
  }
  Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_16U);

  circle(mask, cv::Point(img.cols / 2, img.rows / 2), 2190, cv::Scalar(1), -1);
  cv::Mat temp = mask.mul(img);
  imwrite("mask.png", temp);
  //imwrite("img.png", img);
  return temp;
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
    Mat new_combined_z_buffer = cv::Mat::zeros(cv::Size(new_combined.cols, new_combined.rows), CV_16U);
    
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
  /*
  Mat mask = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);
  
  circle(mask, cv::Point(image_size.width/2, image_size.height/2), 2190, cv::Scalar(255), -1);
  */

  for (int i = 0; i < images.size(); i++){
    //calculate where the new image will be copied to in the composite
    Rect copyzone = Rect(new_info[i].vec.x - root_offset.x, new_info[i].vec.y - root_offset.y,                                                     images[i]->width, images[i]->height);
    
    //calculate which pixels of the new image will be copied into the composite
    Mat use_locations = local_quality_score >= composite_z_buffer(copyzone);
    
    
    if (countNonZero(use_locations) == 0){
      continue; //not contributing, don't bother loading from disk
    }
    
    images[i]->load_raw_from_disk(true);
    Mat image_Mat = cv::Mat(image_size, CV_8U, images[i]->get_Raw(), Mat::AUTO_STEP);
    cvtColor(image_Mat,image_Mat,COLOR_BayerBG2BGR);
    cv::divide(image_Mat,flat_field,image_Mat,1.0,CV_8U);

    //imwrite(images[i]->get_ImageFile().getBaseName()+".png", image_Mat);
    /*
    Mat temp;
    composite_z_buffer.copyTo(temp, copyzone);
        */

    /*
    Mat temp = cv::Mat::zeros(cv::Size(image_size.width, image_size.height), CV_8U);
    image_Mat.copyTo(temp,use_locations);
    imwrite(images[i]->get_ImageFile().getBaseName() + ".png", temp);
    */

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

