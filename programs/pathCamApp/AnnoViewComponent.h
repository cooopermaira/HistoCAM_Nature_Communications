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
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) {
    ImageViewComponent::keyPressed(key, originatingComponent);
    
    return false;  // Key press not handled
  }

  
private:
  std::unique_ptr<AnnotateOverlay> annotateOverlay;
  
  
  double calculatePolygonArea(const std::vector<juce::Point<float>>& vertices) {
      unsigned int n = (unsigned int)vertices.size();
      double area = 0.0;

      // Calculate the area using the shoelace formula
      for (unsigned int i = 0; i < n; i++) {
        unsigned int j = (i + 1) % n; // Wrap around using modulo for the last point
          area += vertices[i].x * vertices[j].y;
          area -= vertices[j].x * vertices[i].y;
      }

      return std::abs(area / 2.0); // Return the absolute value of the area divided by 2
  }
  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnoViewComponent)
  
};


#endif /* AnnoViewComponent_h */
