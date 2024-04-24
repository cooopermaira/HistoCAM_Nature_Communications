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
    repaint();
  }
  if (button == segmentButton.get())
  {
    parent->changeMode( AnnotateComponent::_SEG);
    repaint();
  }
  if (button == dictateButton.get())
  {
    parent->changeMode( AnnotateComponent::_DICT);
    repaint();
  }
  if (button == measureButton.get())
  {
    parent->changeMode( AnnotateComponent::_MEAS);
    repaint();
  }
}

void AnnotateOverlay::paint (juce::Graphics& g)
{
  Component::paint(g);
  g.setColour(juce::Colours::red);
  
  if(parent->getMode() == AnnotateComponent::_POLY){
    g.drawRect(polygonButton->getBounds());
  }
  if(parent->getMode() == AnnotateComponent::_SEG){
    g.drawRect(segmentButton->getBounds());
  }
  if(parent->getMode() == AnnotateComponent::_MEAS){
    g.drawRect(measureButton->getBounds());
  }
  if(parent->getMode() == AnnotateComponent::_DICT){
    g.drawRect(dictateButton->getBounds());
  }
}
