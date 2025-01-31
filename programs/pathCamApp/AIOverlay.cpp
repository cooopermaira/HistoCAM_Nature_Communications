//
//  CaptureOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"


void AIOverlay::resized()
{
  auto area = getLocalBounds().reduced(4);
  
  AIthinkingButton->setVisible(false);
  AIreadyButton->setVisible(false);
    
  if(parent->sCam){
    bool compositing = parent->sCam->compositing;
    bool done = parent->sCam->tileEmbeddingComplete;
    
    if(compositing && !done){
      AIthinkingButton->setVisible(true);
      AIthinkingButton->setBounds(area);
    }
    
    if(compositing && done){
      AIreadyButton->setVisible(true);
      AIreadyButton->setBounds(area);
    }
  }
}

void AIOverlay::paint(juce::Graphics &g) {
  if(AIthinkingButton->isVisible()){
    setAlpha(currentOpacity);
  }
  Component::paint(g);
}


void AIOverlay::buttonClicked(juce::Button* button)
{
  if (button == AIreadyButton.get())
  {
    std::cout << "DO SOMETHING\n";
  }

  resized();
}


void AIOverlay::timerCallback()
{
  // Update the opacity value to create the fade animation
  const float speed = 0.02f; // Change rate per frame

  if (increasingOpacity)
  {
    currentOpacity += speed;
    if (currentOpacity >= 1.0f)
    {
      currentOpacity = 1.0f;
      increasingOpacity = false;
    }
  }
  else
  {
    currentOpacity -= speed;
    if (currentOpacity <= 0.2f)
    {
      currentOpacity = 0.2f;
      increasingOpacity = true;
    }
  }
  
  repaint(); // Trigger a repaint to show the updated opacity
}
