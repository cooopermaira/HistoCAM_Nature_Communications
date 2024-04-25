//
//  AnnoViewComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"


void AnnoViewComponent::changeMode(int mode){ parent->changeMode(mode); }
int AnnoViewComponent::getMode(){ return parent->getMode(); }


void AnnoViewComponent::mouseDown(const juce::MouseEvent& event)
{
  ImageViewComponent::mouseDown(event);
  
  {
    const ScopedLock lock (mutex);
    if(parent->getMode() == AnnotateComponent::_POLY && event.mods.isRightButtonDown()){
      if(selected == NULL){
        std::shared_ptr < Annotation > new_annotation;
        new_annotation.reset(new PolygonAnnotation("Polygon"));
        selected = new_annotation.get();
        annotations->push_back(new_annotation);
        parent->annotationsUpdated();
      }
      PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(selected);
      if(cast == NULL){
        std::shared_ptr < Annotation > new_annotation;
        new_annotation.reset(new PolygonAnnotation("Polygon"));
        annotations->push_back(new_annotation);
        selected = new_annotation.get();
        PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(selected);
        parent->annotationsUpdated();
      }
      
      cast->add( fPoint(event.x, event.y)*screen2view()+view->getPosition() );
    }
    repaint();
  }
}
