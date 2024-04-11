//
//  TiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#include "JuceHeader.h"


std::vector < TileQueryElem >  TiledImage::getTiles(pCApp::Rectangle box){
  std::vector < TileQueryElem > box_tiles;
  
  pCApp::Point top_left = box.getTopLeft();
  pCApp::Point bottom_right = box.getBottomRight();
  bottom_right.x--;
  bottom_right.y--;

  
  for(pCApp::cType i = getIJ(top_left).getX(); i <=  getIJ(bottom_right).getX(); i++){
    for(pCApp::cType j = getIJ(top_left).getY(); j <=  getIJ(bottom_right).getY(); j++){
      box_tiles.push_back(TileQueryElem(tiles(i,j), i, j));
    }
  }
  
  return box_tiles;
}

void TiledImage::insertMat(cv::Mat image_in, pCApp::Rectangle box){
  unsigned int width  = image_in.cols;
  unsigned int height = image_in.rows;
  
  jassert(box.getWidth() == width && box.getHeight() == height);
  
  bounds = bounds.getUnion(box);
    
  pCApp::Point top_left = box.getTopLeft();
  pCApp::Point bottom_right = box.getBottomRight();
  bottom_right.x -= 1;
  bottom_right.y -= 1;
    
  
  for(pCApp::cType i = getIJ(top_left).getX(); i <=  getIJ(bottom_right).getX(); i++){
    for(pCApp::cType j = getIJ(top_left).getY(); j <=  getIJ(bottom_right).getY(); j++){
      
      if(tiles(i,j) == NULL){
        tiles(i,j) = new juce::Image(juce::Image::PixelFormat::RGB, tile_size, tile_size, true);
      }
      
      pCApp::Rectangle tile_box = pCApp::Rectangle(i*tile_size,
                                                   j*tile_size,
                                                   tile_size,
                                                   tile_size);
      
      pCApp::Rectangle image_box = tile_box.getIntersection(box);

      
      pCApp::Point offset = box.getTopLeft();
      
      matToImage(image_in, tiles(i,j), offset, image_box, tile_box);

      
    }
  }
  

}


void TiledImage::matToImage(const cv::Mat& mat, juce::Image *image, 
                            pCApp::Point offset, pCApp::Rectangle image_box,
                            pCApp::Rectangle tile_box){
  jassert(mat.type() == CV_8UC3); // Ensure the Mat is of type 8-bit Unsigned with 3 Channels
  
  cv::Rect ROIrect ((pCApp::cType)(image_box.getX()-offset.getX()), (pCApp::cType)(image_box.getY()-offset.getY()),
                    (pCApp::cType)image_box.getWidth(), (pCApp::cType)image_box.getHeight());

  cv::Mat ROI = mat(ROIrect);
  
  
  juce::Image::BitmapData data(*image, juce::Image::BitmapData::writeOnly);
    
  for (pCApp::cType y = 0, v=(pCApp::cType)(image_box.getY()-tile_box.getY())  ; y < ROI.rows; ++y, ++v) {
      //const auto* matRowPtr = ROI.ptr<cv::Vec3b>(y);
        for (pCApp::cType x = 0, u=(pCApp::cType)(image_box.getX()-tile_box.getX()); x < ROI.cols; ++x, ++u) {
          //const cv::Vec3b& bgr = matRowPtr[x];
          jassert(u < (tile_size) and v < (tile_size));
          cv::Vec3b color = ROI.at<cv::Vec3b>(y, x);
          uint8 alpha = (color[2]==0 &&  color[1]==0 && color[0]==0)? 0:255;

          data.setPixelColour(u, v, juce::Colour(color[2],  color[1] , color[0], alpha));
        }
    }
  
}
