//
//  AnnotateOverlay.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"


void AnnotateOverlay::buttonClicked(juce::Button* button)
{

  if (button == polygonButton.get())
  {
    parent->changeMode( AnnotateComponent::_POLY);
  }
  if (button == segmentButton.get())
  {
    parent->changeMode( AnnotateComponent::_SEG);
  }
  if (button == dictateButton.get())
  {
    parent->changeMode( AnnotateComponent::_DICT);
  }
  if (button == measureButton.get())
  {
    parent->changeMode( AnnotateComponent::_MEAS);
  }
}
