//
//  SvgButton.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/17/24.
//

#ifndef SvgButton_h
#define SvgButton_h

#include "JuceHeader.h"

class SvgButton : public juce::Button
{
public:
    SvgButton(const juce::String& buttonName, const juce::Drawable * svgDrawable) : juce::Button(buttonName)
    {
        if (svgDrawable != nullptr)
        {
            drawable = std::unique_ptr<juce::Drawable>(svgDrawable->createCopy());
        }
    }

    void paintButton(juce::Graphics& g, bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override
    {
        if (drawable != nullptr)
        {
            auto bounds = getLocalBounds().toFloat();
            drawable->drawWithin(g, bounds, juce::RectanglePlacement::centred, 1.0f);

            if (shouldDrawButtonAsDown)
            {
                g.setColour(juce::Colours::white.withAlpha(0.3f));  // semi-transparent white overlay
                g.fillRect(bounds);
            }
        }
    }

private:
    std::unique_ptr<juce::Drawable> drawable;
};


#endif /* SvgButton_hpp */
