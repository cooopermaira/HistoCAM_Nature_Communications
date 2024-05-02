//
//  ColorButton.h
//  pathCam
//
//  Created by Brian Summa on 5/2/24.
//

#ifndef ColorButton_h
#define ColorButton_h

#include "JuceHeader.h"

class ColorButton : public juce::Button
{
public:
  ColorButton(const juce::String& buttonName) : juce::Button(buttonName) { }
  
  void paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override
  {
    auto bounds = getLocalBounds().toFloat();
    if (shouldDrawButtonAsDown)
    {
      g.setColour(juce::Colours::white.withAlpha(0.3f));  // semi-transparent white overlay
    }else{
      g.setColour(color);
    }
    
    g.fillRect(bounds);
  }
  
  void setColor(juce::Colour new_color){ color = new_color; }
  juce::Colour getColor(){ return color; }
  
private:
  juce::Colour color;
};


#endif /* ColorButton_hpp */
