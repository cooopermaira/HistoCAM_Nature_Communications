//
//  Annotation.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"



int Annotation::getType(){
  {
    PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(this);
    if(cast != NULL){ return _POLY; }
  }
  {
    DictateAnnotation *cast = dynamic_cast < DictateAnnotation * >(this);
    if(cast != NULL){ return _DICT; }
  }
  {
    SegmentAnnotation *cast = dynamic_cast < SegmentAnnotation * >(this);
    if(cast != NULL){ return _SEG; }
  }
  {
    MeasureAnnotation *cast = dynamic_cast < MeasureAnnotation * >(this);
    if(cast != NULL){ return _MEAS; }
  }

  return _NONE;
}
