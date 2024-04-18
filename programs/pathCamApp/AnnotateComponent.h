//
//  AnnotateComponent.h
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef AnnotateComponent_h
#define AnnotateComponent_h

#include "JuceHeader.h"

class AnnotateComponent: public ImageViewComponent{
public:
  AnnotateComponent(MainComponent *parent, 
                    std::shared_ptr < fRectangle > view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile) : ImageViewComponent(parent,view,iconNames,iconsFromZipFile), recording(false){

    annotateOverlay.reset (new AnnotateOverlay (this, iconNames, iconsFromZipFile));
    addAndMakeVisible (annotateOverlay.get());

                     
  }
  
  void resized()
  {
    
    ImageViewComponent::resized();

    {
      const ScopedLock lock (mutex);
      juce::Rectangle<int> b = getLocalBounds();
      int width = 300;
      annotateOverlay->setBounds(juce::Rectangle<int>(b.getWidth()-width-20, 20, width, 60));
      
    }
    
  }


  void paint (juce::Graphics& g)
  {
    ImageViewComponent::paint(g);
    
  }

private:
  std::unique_ptr<AnnotateOverlay> annotateOverlay;
  
  bool recording;
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnotateComponent)

};

#endif /* AnnotateComponent_h */
