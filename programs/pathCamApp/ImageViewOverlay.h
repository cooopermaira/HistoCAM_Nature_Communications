//
//  ImageViewOverlay.hpp
//  pathCam
//
//  Created by Brian Summa on 4/16/24.
//

#ifndef ImageViewOverlay_h
#define ImageViewOverlay_h

#include "JuceHeader.h"


class ImageViewOverlay final : public Component,
                               public Button::Listener
{
public:
  ImageViewOverlay ()
    {
      addAndMakeVisible(myButton);
      myButton.addListener(this);
      

    }

    void resized() override
    {
        auto area = getLocalBounds().reduced (4);
        myButton.setBounds(area);
    }

private:
  
  void buttonClicked(juce::Button* button) override
  {
      if (button == &myButton)
      {
          juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::InfoIcon,
                                                 "Button Clicked",
                                                 "You clicked the button!");
      }
  }

  TextButton myButton { "Click Me" };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (ImageViewOverlay)
};


#endif /* ImageViewOverlay_hpp */
