//
//  ImageViewOverlay.hpp
//  pathCam
//
//  Created by Brian Summa on 4/16/24.
//

#ifndef ImageViewOverlay_h
#define ImageViewOverlay_h

#include "JuceHeader.h"

class ImageViewComponent;

class ImageViewOverlay final : public Component,
                               public Button::Listener {
public:
  ImageViewOverlay(ImageViewComponent *parent,
                   StringArray &iconNames,
                   OwnedArray<Drawable> &iconsFromZipFile) : parent(parent) {
    for (int i = 0; i < iconNames.size(); i++) {
      auto name = iconNames[i];
      if (iconNames[i] == "center.svg") {
        centerButton.reset(new SvgButton("center", iconsFromZipFile[i]));
        centerButton->addListener(this);
        //break;
      }
      if (iconNames[i] == "slideList.svg") {
        slideListButton.reset(new SvgButton("slides", iconsFromZipFile[i]));
        slideListButton->addListener(this);
        addAndMakeVisible(*slideListButton);
      }
    }

    addAndMakeVisible(*centerButton);
  }

  ~ImageViewOverlay() {
  }

  void resized() override {
    auto area = getLocalBounds().reduced(8);

    auto top = area.removeFromTop(area.getHeight() / 2);
    auto bottom = area;

    centerButton->setBounds(top);
    slideListButton->setBounds(bottom);
  }

private:
  void buttonClicked(juce::Button *button) override;

  std::unique_ptr<SvgButton> centerButton;
  std::unique_ptr<SvgButton> slideListButton;

  ImageViewComponent *parent;


  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ImageViewOverlay)
};


#endif /* ImageViewOverlay_hpp */
