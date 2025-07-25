//
//  AnnotateComponent.h
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef AnnotateComponent_h
#define AnnotateComponent_h

#include "JuceHeader.h"

class AnnotateComponent  : public juce::Component {
    
public:
  MainComponent* parent;

  AnnotateComponent(std::shared_ptr < fRectangle > view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile,
                    MainComponent* parent): parent(parent), mode(Annotation::_NONE) {
    
    annotations.reset( new std::vector < std::shared_ptr<  Annotation > >());
    
    leftComponent.reset( new AnnoListComponent(this, annotations, iconNames, iconsFromZipFile) );
    
    addChildComponent(leftComponent.get());
    
    rightComponent.reset( new AnnoViewComponent(this, view, iconNames, iconsFromZipFile, annotations));
    
    addChildComponent(rightComponent.get());
    
    layout.setItemLayout(0, 100, -1, -0.20);  // left component takes half the space initially
    layout.setItemLayout(1, 10, 10, 10);    // resizer bar with a fixed size
    layout.setItemLayout(2, 100, -1, -0.80);  // right component also takes half the space initially
    
    resizerBar.reset(new juce::StretchableLayoutResizerBar(&layout, 1, true));
    addChildComponent(resizerBar.get());
    /*
#if DEBUG
    {
      std::shared_ptr< PolygonAnnotation > temp = std::make_shared < PolygonAnnotation >("Poly 1");
      
      temp->add(fPoint(50, 50));
      temp->add(fPoint(1200, 150));          // Add second point
      temp->add(fPoint(1150, 1150));         // Add third point
      temp->add(fPoint(150, 1100));          // Add fourth point
      temp->add(fPoint(-150, 500));          // Add fifth point
      
      annotations->push_back(temp);
    }
    {
      std::shared_ptr< PolygonAnnotation > temp = std::make_shared < PolygonAnnotation >("Poly 2");
      
      temp->add(fPoint(2050, 2050));
      temp->add(fPoint(2050, 3050));          // Add second point
      temp->add(fPoint(3050, 3050));         // Add third point
      temp->add(fPoint(3050, 2050));          // Add fourth point
      
      annotations->push_back(temp);
    }
    
    {
      std::shared_ptr< MeasureAnnotation > temp = std::make_shared < MeasureAnnotation >("Measure 1");
      temp->setPoints(fPoint(1000, 1000), fPoint(2000,1000));
      annotations->push_back(temp);
    }
    
    {
      std::shared_ptr< MeasureAnnotation > temp = std::make_shared < MeasureAnnotation >("Measure 2");
      temp->setPoints(fPoint(4000, 4000), fPoint(5000,5000));
      annotations->push_back(temp);
    }
    
    
    
    leftComponent->updatelist();
#endif
*/
  }
  
  void setImage(std::shared_ptr<MRTiledImageSet > image){  rightComponent->setImage(image); }
  
  void refreshImage(){ rightComponent->refreshImage(); }
  
  void resized() override  {
    auto area = getLocalBounds();
    juce::Component* components[] = { leftComponent.get(), resizerBar.get(), rightComponent.get() };

    layout.layOutComponents(components, 3, area.getX(), area.getY(), area.getWidth(), area.getHeight(), false, true);
    rightComponent->resized();
  }
  
  void setVisible (bool shouldBeVisible) override {
    Component::setVisible(shouldBeVisible);
    
    leftComponent->setVisible(shouldBeVisible);
    rightComponent->setVisible(shouldBeVisible);
    resizerBar->setVisible(shouldBeVisible);

  }
  
  void fixAspectRatio(){ rightComponent->fixAspectRatio(); }
  
  AnnoViewComponent * getViewComp(){ return rightComponent.get(); }
  AnnoListComponent * getListComp(){ return leftComponent.get(); }
  
  void toggleMode(int mode_in){
    if(mode_in == mode){
      mode=Annotation::_NONE;
    }else{
      mode=mode_in;
    }
  }
  
  void setMode(int mode_in){ mode = mode_in; }

  
  int getMode(){ return mode; }
  
  void annotationsUpdated(){ leftComponent->updatelist(); }
  
  std::shared_ptr< Annotation > getSelected(){ return selected; }
  void setSelected(std::shared_ptr< Annotation > annotation){ selected = annotation; }
  
  void removeSelected(){
    auto it = std::find(annotations->begin(), annotations->end(), selected);
    annotations->erase(it);
    selected = NULL;
    leftComponent->updatelist();
    repaint();
  }

private:
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  std::shared_ptr< Annotation > selected;
  
  std::unique_ptr< AnnoListComponent > leftComponent;
  std::unique_ptr< AnnoViewComponent > rightComponent;
  std::unique_ptr<juce::StretchableLayoutResizerBar> resizerBar;
  juce::StretchableLayoutManager layout;


  
  int mode;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnotateComponent)

};

#endif /* AnnotateComponent_h */
