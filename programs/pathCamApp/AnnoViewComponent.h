//
//  AnnoViewComponent.h
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef AnnoViewComponent_h
#define AnnoViewComponent_h

#include "JuceHeader.h"

class AnnoViewComponent  : public ImageViewComponent {
public:
  AnnoViewComponent(MainComponent *parent,
                    std::shared_ptr < fRectangle > view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile): ImageViewComponent(parent,view,iconNames,iconsFromZipFile)
  {
                      
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
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnoViewComponent)
  
};


#endif /* AnnoViewComponent_h */
