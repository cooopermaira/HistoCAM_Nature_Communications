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
    parent->toggleMode( Annotation::_POLY);
    parent->setSelected(NULL);
    repaint();
  }
  if (button == segmentButton.get())
  {
    parent->toggleMode( Annotation::_SEG);
    parent->setSelected(NULL);
    repaint();
  }
  if (button == dictateButton.get())
  {
    parent->toggleMode( Annotation::_DICT);
    parent->setSelected(NULL);
    repaint();
  }
  if (button == measureButton.get())
  {
    parent->toggleMode( Annotation::_MEAS);
    parent->setSelected(NULL);
    repaint();
  }
}

void AnnotateOverlay::paint (juce::Graphics& g)
{
  Component::paint(g);
  g.setColour(juce::Colours::red);
  
  if(parent->getMode() == Annotation::_POLY){
    g.drawRect(polygonButton->getBounds());
  }
  if(parent->getMode() == Annotation::_SEG){
    g.drawRect(segmentButton->getBounds());
  }
  if(parent->getMode() == Annotation::_MEAS){
    g.drawRect(measureButton->getBounds());
  }
  if(parent->getMode() == Annotation::_DICT){
    g.drawRect(dictateButton->getBounds());
  }
}
