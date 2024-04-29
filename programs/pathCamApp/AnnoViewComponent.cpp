//
//  AnnoViewComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"


void AnnoViewComponent::toggleMode(int mode){ parent->toggleMode(mode); }
void AnnoViewComponent::changeMode(int mode){ parent->changeMode(mode); }

int AnnoViewComponent::getMode(){ return parent->getMode(); }

void AnnoViewComponent::polyMouseDown(const juce::MouseEvent& event)
{
  
  if(parent->getSelected() == NULL){
    std::shared_ptr < Annotation > new_annotation;
    new_annotation.reset(new PolygonAnnotation("Polygon"));
    parent->setSelected(new_annotation);
    annotations->push_back(new_annotation);
    parent->annotationsUpdated();
  }
  PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
  if(cast == NULL){
    std::shared_ptr < Annotation > new_annotation;
    new_annotation.reset(new PolygonAnnotation("Polygon"));
    annotations->push_back(new_annotation);
    parent->setSelected(new_annotation);
    PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
    parent->annotationsUpdated();
  }
  
  cast->add( fPoint(event.x, event.y)*screen2view()+view->getPosition() );
  
  repaint();
}

void AnnoViewComponent::polyMouseDrag(const juce::MouseEvent& event)
{
  
}

void AnnoViewComponent::mouseDown(const juce::MouseEvent& event)
{
  ImageViewComponent::mouseDown(event);
  
  {
    const ScopedLock lock (mutex);
    if(parent->getMode() != AnnotateComponent::_POLY && event.mods.isRightButtonDown()){
      polyMouseDown(event);
    }
    
  }
}

void AnnoViewComponent::mouseDrag(const juce::MouseEvent& event){
  ImageViewComponent::mouseDrag(event);

  {
    const ScopedLock lock (mutex);
    if(parent->getMode() != AnnotateComponent::_POLY && event.mods.isRightButtonDown()){
      polyMouseDrag(event);
    }
    
  }
  
}


void AnnoViewComponent::paint (juce::Graphics& g)
{
  ImageViewComponent::paint(g);
  
  {
    const ScopedLock lock (mutex);
    for(unsigned int i=0; i < annotations->size(); i++){
      (*annotations)[i]->paint(g, view->getPosition(), (*annotations)[i].get() == parent->getSelected().get(), view2screen());
    }
  }
}
