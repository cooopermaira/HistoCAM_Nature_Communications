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
class MainComponent;

class AnnoViewComponent  : public ImageViewComponent {
public:
  AnnoViewComponent(AnnotateComponent *parent,
                    std::shared_ptr < fRectangle > view,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile,
                    std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations);
  
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
  
  
  void paint (juce::Graphics& g) override;
  
  bool keyPressed(const juce::KeyPress& key, juce::Component* originatingComponent) override;
  
  void mouseDown(const juce::MouseEvent& event) override;
  void mouseUp(const juce::MouseEvent& event) override;
  void mouseDrag(const juce::MouseEvent& event) override;
  void mouseMove(const juce::MouseEvent& event) override;
  
  bool polyMouseDown(const juce::MouseEvent& event);
  bool polyMouseUp(const juce::MouseEvent& event);
  bool polyMouseDrag(const juce::MouseEvent& event);

  bool measureMouseDown(const juce::MouseEvent& event);
  bool measureMouseMove(const juce::MouseEvent& event);

  bool segMouseDown(const juce::MouseEvent& event);
  bool segMouseUp(const juce::MouseEvent& event);
  bool segMouseDrag(const juce::MouseEvent& event);

  void toggleMode(int mode);
  void setMode(int mode);
  int getMode();
  void setSelected(std::shared_ptr< Annotation > annotation);

  
  std::unique_ptr<AnnotateOverlay> annotateOverlay;
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
    
  AnnotateComponent *annotateParent;

  
  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AnnoViewComponent)
  
};


#endif /* AnnoViewComponent_h */
