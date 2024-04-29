//
//  AnnoViewComponent.h
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef AnnoViewComponent_h
#define AnnoViewComponent_h

#include "JuceHeader.h"

class AnnotateComponent;

class AnnoViewComponent  : public ImageViewComponent {
public:
  AnnoViewComponent(AnnotateComponent *parent,
                    std::shared_ptr < fRectangle > view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile,
                    std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations): ImageViewComponent(view,iconNames,iconsFromZipFile), parent(parent), annotations(annotations)
  {
      
      annotateOverlay.reset (new AnnotateOverlay (this, iconNames, iconsFromZipFile));
      addAndMakeVisible (annotateOverlay.get());
                                       
  }
  
  void resized() override
  {
    
    ImageViewComponent::resized();
    
    {
      const ScopedLock lock (mutex);
      juce::Rectangle<int> b = getLocalBounds();
      int width = 400;
      annotateOverlay->setBounds(juce::Rectangle<int>(b.getWidth()-width-20, 20, width, 60));
      
    }
    
  
    
  }
  
  
  void paint (juce::Graphics& g);
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override {
    ImageViewComponent::keyPressed(key, originatingComponent);
    
    return false;  // Key press not handled
  }
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  
  void polyMouseDown(const juce::MouseEvent& event);
  void polyMouseDrag(const juce::MouseEvent& event);


  void toggleMode(int mode);
  void changeMode(int mode);
  int getMode();
  
  
private:
  std::unique_ptr<AnnotateOverlay> annotateOverlay;
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
    
  AnnotateComponent *parent;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnoViewComponent)
  
};


#endif /* AnnoViewComponent_h */
