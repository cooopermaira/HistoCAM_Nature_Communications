//
//  ImageList.h
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#ifndef ImageList_h
#define ImageList_h

#include "common.h"


namespace pathCam{
  
  class ImageList{
  public:
    std::vector < Image *> images;
    
    ImageList(){};
    ~ImageList(){
      for(unsigned int i=0; i < images.size(); i++){
        delete images[i];
      }
      images.clear();
    }
    
    bool loadFileList(std::string filelist);
  };
  
}

#endif /* ImageList_h */
