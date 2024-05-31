//
//  Composite.h
//  pathCamLib
//
//  Created by cooper maira on 12/23/23.
//

#ifndef Composite_h
#define Composite_h

#include <stdio.h>
#include "pathCam.h"
#include "JuceHeader.h"
#include "TiledImage.h"

namespace pathCam{

class Composite{
	friend class CompositeManager;
protected:
  StreamCam *parent;
  Mat local_quality_score, composite_z_buffer, flat_field;
  Mat3f flat_field_composite;
  Mat3b composite;
  Vec2 root_offset,max_offset;
  Bbox subdiv_Bbox;
  fRectangle tiledImageBounds;
  
  
public:
  Composite(StreamCam *parent);
  
  void add_images(std::vector < RegInfo > new_info);
  void update_Bbox(std::vector < RegInfo > new_info);
  void update(std::vector < RegInfo > new_info);
  
  Mat get_composite();
  Mat score_image_2X(int,int,int);
};


class CompositeVoronoi : public Composite {
private:
	cv::Subdiv2D subdiv;
    std::vector<std::pair<std::string,bool>> memberImages;
    Mat circleMask;
    cv::Size image_size;


public:
	CompositeVoronoi(StreamCam* parent,cv::Size image_size);
	void update(std::vector < RegInfo > new_info);
	void add_images(std::vector < RegInfo > new_info);
	void expand_subdiv(std::vector < RegInfo > new_info);

};
}

#endif /* Composite_h */
