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
  AnnotateComponent(MainComponent *parent,
                    std::shared_ptr < fRectangle > view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile) {

    annotations.reset( new std::vector < std::shared_ptr<  Annotation > >());
    
    leftComponent.reset( new AnnoListComponent(annotations) );

    addChildComponent(leftComponent.get());
    
    rightComponent.reset( new AnnoViewComponent(parent, this, view, iconNames, iconsFromZipFile, annotations));
    
    addChildComponent(rightComponent.get());

    layout.setItemLayout(0, 100, -1, 0.5);  // left component takes half the space initially
    layout.setItemLayout(1, 10, 10, 10);    // resizer bar with a fixed size
    layout.setItemLayout(2, 100, -1, 0.5);  // right component also takes half the space initially

    resizerBar.reset(new juce::StretchableLayoutResizerBar(&layout, 1, true));
    addChildComponent(resizerBar.get());
    
#if DEBUG
    
    std::shared_ptr< PolygonAnnotation > temp = std::make_shared < PolygonAnnotation >("Poly 1");
    
    temp->path.startNewSubPath(50, 50);
    temp->path.lineTo(200, 50);          // Add second point
    temp->path.lineTo(150, 150);         // Add third point
    temp->path.lineTo(50, 100);          // Add fourth point
    temp->path.closeSubPath();           // Close the path
  
    annotations->push_back(temp);
    
        
#endif
  }
  
  void setImage(std::shared_ptr<MRTiledImage> image){  rightComponent->setImage(image); }
  
  void resized()
  {
    if(!isVisible()){ return; }
    auto area = getLocalBounds();
    juce::Component* components[] = { leftComponent.get(), resizerBar.get(), rightComponent.get() };

//    // This will position and resize the components according to the layout
    layout.layOutComponents(components, 3, area.getX(), area.getY(), area.getWidth(), area.getHeight(), false, true);

  }
  
  AnnoViewComponent * getViewComp(){ return rightComponent.get(); }


private:
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  
  std::unique_ptr< AnnoListComponent > leftComponent;
  std::unique_ptr< AnnoViewComponent > rightComponent;
  std::unique_ptr<juce::StretchableLayoutResizerBar> resizerBar;
  juce::StretchableLayoutManager layout;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnotateComponent)

};

#endif /* AnnotateComponent_h */
