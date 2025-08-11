//
//  AnnoViewComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"

AnnoViewComponent::AnnoViewComponent(AnnotateComponent *parent, std::shared_ptr<fRectangle> view,
                                     juce::StringArray &iconNames, OwnedArray<juce::Drawable> &iconsFromZipFile,
                                     std::shared_ptr<std::vector<std::shared_ptr<Annotation>>> annotations) : ImageViewComponent(view,iconNames,iconsFromZipFile,parent->parent), parent(parent), annotations(annotations)
{

  annotateOverlay.reset (new AnnotateOverlay (this, iconNames, iconsFromZipFile));
  addAndMakeVisible (annotateOverlay.get());

}

void AnnoViewComponent::toggleMode(int mode){ parent->toggleMode(mode); }
void AnnoViewComponent::setMode(int mode){ parent->setMode(mode); }

int AnnoViewComponent::getMode(){ return parent->getMode(); }
void AnnoViewComponent::setSelected(std::shared_ptr< Annotation > annotation){ parent->setSelected(annotation); }


bool AnnoViewComponent::keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) {
  ImageViewComponent::keyPressed(key, originatingComponent);

  if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_SEG) {
    if (key == juce::KeyPress::createFromDescription("s")) {
      std::cout << "Run SAM\n";
      SegmentAnnotation *cast = dynamic_cast < SegmentAnnotation * >(parent->getSelected().get());
      //cast->call_SAM();
      parent->parent->sCam->segment_with_SAM(cast->input,parent->segID++);
      repaint();
      return true; // Key press handled
    }
  }



  return false;  // Key press not handled
}

bool AnnoViewComponent::polyMouseDown(const juce::MouseEvent& event)
{
  if(event.mods.isRightButtonDown()){
    if(parent->getMode() == Annotation::_POLY){
      std::shared_ptr < Annotation > new_annotation;
      new_annotation.reset(new PointClickPoly("Polygon"));
      parent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      parent->annotationsUpdated();
      parent->setMode( Annotation::_NONE );
    }
    
    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
      PointClickPoly *cast = dynamic_cast < PointClickPoly * >(parent->getSelected().get());
      cast->add(screen2view(fPoint(event.x, event.y), *view));
      return true;
    }

    /*if(parent->getMode() == Annotation::_SEG){
      std::shared_ptr < Annotation > new_annotation;
      new_annotation.reset(new PolygonAnnotation("SAM"));
      parent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      parent->annotationsUpdated();
      parent->setMode( Annotation::_NONE );
    }

    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_SEG){
      PolygonAnnotation *cast = dynamic_cast < PolygonAnnotation * >(parent->getSelected().get());
      cast->add(screen2view(fPoint(event.x, event.y), *view));
      return true;
    }*/
  }
  
  
  if(event.mods.isLeftButtonDown()){
    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
      PointClickPoly *cast = dynamic_cast < PointClickPoly * >(parent->getSelected().get());
      if(cast->test(screen2view(fPoint(event.x, event.y), *view), screen2viewScale(*view))){
        return true;
      }
    }
  }

  return false;
}

bool AnnoViewComponent::polyMouseDrag(const juce::MouseEvent& event)
{
  if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
    PointClickPoly *cast = dynamic_cast < PointClickPoly * >(parent->getSelected().get());
    if(cast->isPointSelected()){
      cast->move(screen2view(fPoint(event.x, event.y), *view));
      return true;
    }
  }
  return false;
}

bool AnnoViewComponent::polyMouseUp(const juce::MouseEvent& event){
  if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_POLY){
    int q = parent->getSelected()->getType();
    PointClickPoly *cast = dynamic_cast < PointClickPoly * >(parent->getSelected().get());
    cast->unSelect();
    return true;
  }
  
  return false;
}

bool AnnoViewComponent::measureMouseDown(const juce::MouseEvent& event){
  if(event.mods.isRightButtonDown()){
    if(parent->getMode() == Annotation::_MEAS){
      std::shared_ptr < Annotation > new_annotation;
      new_annotation.reset(new MeasureAnnotation("Measure"));
      parent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      parent->annotationsUpdated();
      parent->setMode( Annotation::_NONE );
    }
    
    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_MEAS){
      MeasureAnnotation *cast = dynamic_cast < MeasureAnnotation * >(parent->getSelected().get());
      if(cast->isMeasuring()){
        cast->stopMeasuring(screen2view(fPoint(event.x, event.y), *view));
      }else{
        cast->startMeasuring(screen2view(fPoint(event.x, event.y), *view));
      }
      return true;
    }
  }
return false;
}

bool AnnoViewComponent::measureMouseMove(const juce::MouseEvent& event){
  if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_MEAS){
    MeasureAnnotation *cast = dynamic_cast < MeasureAnnotation * >(parent->getSelected().get());
    if(cast->isMeasuring()){
      cast->continueMeasuring(screen2view(fPoint(event.x, event.y), *view));
      return true;
    }
  }
    
  return false;
}

bool AnnoViewComponent::segMouseDown(const juce::MouseEvent& event)
{
  if(event.mods.isLeftButtonDown()){
    if(parent->getMode() == Annotation::_SEG){
      std::shared_ptr < Annotation > new_annotation;
      new_annotation.reset(new SegmentAnnotation("SAM"));
      parent->setSelected(new_annotation);
      annotations->push_back(new_annotation);
      parent->annotationsUpdated();
      parent->setMode( Annotation::_NONE );
    }

    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_SEG){
      SegmentAnnotation *cast = dynamic_cast < SegmentAnnotation * >(parent->getSelected().get());
      fPoint temp = screen2view(fPoint(event.x, event.y), *view);
      cast->add(Point3f(temp.x, temp.y, 1.0));
      return true;
    }
  }

  if(event.mods.isRightButtonDown()){
    if(parent->getSelected() != NULL && parent->getSelected()->getType() == Annotation::_SEG){
      SegmentAnnotation *cast = dynamic_cast < SegmentAnnotation * >(parent->getSelected().get());
      fPoint temp = screen2view(fPoint(event.x, event.y), *view);
      cast->add(Point3f(temp.x, temp.y, 0.0));
      return true;
    }
  }

  return false;
}

bool AnnoViewComponent::segMouseUp(const juce::MouseEvent& event) {
    if (parent->getMode() == Annotation::_SEG || parent->getSelected()->getType() == Annotation::_SEG) {
      return true;
    }
}

bool AnnoViewComponent::segMouseDrag(const juce::MouseEvent& event) {
  if (parent->getMode() == Annotation::_SEG || parent->getSelected()->getType() == Annotation::_SEG) {
    return true;
  }
}

void AnnoViewComponent::mouseDown(const juce::MouseEvent& event) {
  
  bool handled;

  {
    const ScopedLock lock (mutex);
    handled = polyMouseDown(event);
    if(!handled){ handled = measureMouseDown(event); }
    if(!handled){ handled = segMouseDown(event); }
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
    if(!handled){ handled = segMouseUp(event); }
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
    if(!handled){ handled = segMouseDrag(event); }
    if(handled){ repaint();}
  }
  
  if(!handled){
    ImageViewComponent::mouseDrag(event);
  }

  
}

void AnnoViewComponent::mouseMove(const juce::MouseEvent& event){
  
  bool handled;
  
  {
    const ScopedLock lock (mutex);
    handled = measureMouseMove(event);
    if(handled){ repaint();}
  }
  
  if(!handled){
    ImageViewComponent::mouseMove(event);
  }
}

void AnnoViewComponent::paint (juce::Graphics& g)
{
  ImageViewComponent::paint(g);
  
  {
    const ScopedLock lock (mutex);
    for(unsigned int i=0; i < annotations->size(); i++){
      (*annotations)[i]->paint(g, view->getPosition(), (*annotations)[i].get() == parent->getSelected().get(), view2screenScale(*view));
    }
  }
}
