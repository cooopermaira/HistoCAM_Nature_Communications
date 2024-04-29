//
//  AnnoViewComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"


void AnnoViewComponent::toggleMode(int mode){ parent->toggleMode(mode); }
void AnnoViewComponent::setMode(int mode){ parent->setMode(mode); }

int AnnoViewComponent::getMode(){ return parent->getMode(); }

bool AnnoViewComponent::polyMouseDown(const juce::MouseEvent& event)
{
  if(event.mods.isRightButtonDown()){
    if(parent->getMode() == Annotation::_POLY){
      std::shared_ptr < Annotation > new_annotation;
      new_annotation.reset(new PolygonAnnotation("Polygon"));
      parent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      parent->annotationsUpdated();
      parent->setMode( Annotation::_NONE );
    }
    
    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
      PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
      cast->add(screen2view(fPoint(event.x, event.y)));
    }
    return true;
  }
  
  
  if(event.mods.isLeftButtonDown()){
    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
      PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
      if(cast->test(screen2view(fPoint(event.x, event.y)), screen2viewScale())){
        return true;
      }
    }
  }

  return false;
}

bool AnnoViewComponent::polyMouseDrag(const juce::MouseEvent& event)
{
  if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
    PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
    if(cast->isPointSelected()){
      cast->move(screen2view(fPoint(event.x, event.y)));
      return true;
    }
  }
  return false;
}

bool AnnoViewComponent::polyMouseUp(const juce::MouseEvent& event){
  if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
    PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
    cast->unSelect();
    return true;
  }
  
  return false;
}

void AnnoViewComponent::mouseDown(const juce::MouseEvent& event) {
  
  bool handled;

  {
    const ScopedLock lock (mutex);
    handled = polyMouseDown(event);
    if(handled){ repaint(); }
  }
  
  if(!handled){
    ImageViewComponent::mouseDown(event);
  }

  
}

void AnnoViewComponent::mouseUp(const juce::MouseEvent& event) {
  
  bool handled;

  {
    const ScopedLock lock (mutex);
    handled = polyMouseUp(event);
    if(handled){ repaint(); }
  }
  
  if(!handled){
    ImageViewComponent::mouseUp(event);
  }

  
}


void AnnoViewComponent::mouseDrag(const juce::MouseEvent& event){

  bool handled;
  
  {
    const ScopedLock lock (mutex);
    handled = polyMouseDrag(event);
    if(handled){ repaint();}
  }
  
  if(!handled){
    ImageViewComponent::mouseDrag(event);
  }

  
}


void AnnoViewComponent::paint (juce::Graphics& g)
{
  ImageViewComponent::paint(g);
  
  {
    const ScopedLock lock (mutex);
    for(unsigned int i=0; i < annotations->size(); i++){
      (*annotations)[i]->paint(g, view->getPosition(), (*annotations)[i].get() == parent->getSelected().get(), view2screenScale());
    }
  }
}
