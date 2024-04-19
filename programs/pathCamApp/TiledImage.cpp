//
//  TiledImage.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/6/24.
//

#include "JuceHeader.h"


std::vector < TileQuery >  TiledImage::getTiles(fRectangle box){
  std::vector < TileQuery > box_tiles;
  
  fPoint top_left = box.getTopLeft();
  fPoint bottom_right = box.getBottomRight();
  bottom_right.x--;
  bottom_right.y--;

  
  for(int i = getIJ(top_left).getX(); i <=  getIJ(bottom_right).getX(); i++){
    for(int j = getIJ(top_left).getY(); j <=  getIJ(bottom_right).getY(); j++){
      int x = i*(int)logic_size - box.getX();
      int y = j*(int)logic_size - box.getY();
      juce::Rectangle< float > rect = juce::Rectangle< float >(x, y, logic_size, logic_size);
      box_tiles.push_back(TileQuery(tiles(i,j), i, j, rect));
    }
  }
  
  return box_tiles;
}

void TiledImage::insertMat(cv::Mat image_in, fRectangle box){
  unsigned int width  = image_in.cols;
  unsigned int height = image_in.rows;
  float scale = ((float)tile_size/(float)logic_size);
  
  jassert(int(box.getWidth()*scale) == width &&
          int(box.getHeight()*scale) == height);
  
  bounds = bounds.getUnion(box);
    
  fPoint top_left = box.getTopLeft();
  fPoint bottom_right = box.getBottomRight();
  bottom_right.x -= 1;
  bottom_right.y -= 1;
    
  
  for(int i = getIJ(top_left).getX(); i <=  getIJ(bottom_right).getX(); i++){
    for(int j = getIJ(top_left).getY(); j <=  getIJ(bottom_right).getY(); j++){
      
      if(tiles(i,j) == NULL){
        tiles(i,j) = new juce::Image(juce::Image::PixelFormat::ARGB, tile_size, tile_size, true);
      }
      
      fRectangle tile_box = fRectangle(i*logic_size,
                                       j*logic_size,
                                       logic_size,
                                       logic_size);
      
      fRectangle image_box = tile_box.getIntersection(box);

      
      fPoint offset = box.getTopLeft();
      
      matToImage(image_in, tiles(i,j), offset*scale, image_box*scale, tile_box*scale);

      
    }
  }
  

}


void TiledImage::matToImage(const cv::Mat& mat, juce::Image *image, 
                            fPoint offset, fRectangle image_box,
                            fRectangle tile_box){
  jassert(mat.type() == CV_8UC3);
  
  cv::Rect ROIrect ((int)(image_box.getX()-offset.getX()),
                    (int)(image_box.getY()-offset.getY()),
                    (int)image_box.getWidth(),
                    (int)image_box.getHeight());

  cv::Mat ROI = mat(ROIrect);
  
  
  juce::Image::BitmapData data(*image, juce::Image::BitmapData::writeOnly);
    
  for (int y = 0, v=(int)(image_box.getY()-tile_box.getY())  ; y < ROI.rows; ++y, ++v) {
      const auto* matRowPtr = ROI.ptr<cv::Vec3b>(y);
        for (int x = 0, u=(int)(image_box.getX()-tile_box.getX()); x < ROI.cols; ++x, ++u) {
          const cv::Vec3b& bgr = matRowPtr[x];
          jassert(u < (tile_size) and v < (tile_size));
          //cv::Vec3b color = ROI.at<cv::Vec3b>(y, x);
          uint8 alpha = (bgr[2]==0 &&  bgr[1]==0 && bgr[0]==0)? 0:255;

          data.setPixelColour(u, v, juce::Colour(bgr[2],  bgr[1] , bgr[0], alpha));
        }
    }
  
}
