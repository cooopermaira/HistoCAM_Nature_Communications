//
//  ImageViewOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/16/24.
//

#include "JuceHeader.h"

void ImageViewOverlay::buttonClicked(juce::Button *button) {
  if (button == centerButton.get()) {
    parent->zoomAndCenter();
    parent->repaint();
  }
}


