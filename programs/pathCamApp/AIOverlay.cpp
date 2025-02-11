//
//  CaptureOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"
AIOverlay::  AIOverlay(CaptureComponent *parent,
                       StringArray &iconNames,
                       OwnedArray<Drawable> &iconsFromZipFile) : parent(parent) {
  AIready = true;


  for (int i = 0; i < iconNames.size(); i++) {

    if (iconNames[i] == "AI_ready.svg") {
      AIreadyButton.reset(new SvgButton("AI_ready", iconsFromZipFile[i]));
      AIreadyButton->addListener(this);
      addAndMakeVisible(*AIreadyButton);
    }

    if (iconNames[i] == "AI_thinking.svg") {
      AIthinkingButton.reset(new SvgButton("AI_thinking", iconsFromZipFile[i]));
      AIthinkingButton->addListener(this);
      addAndMakeVisible(*AIthinkingButton);
    }

  }

  startTimerHz(30);

}

void AIOverlay::resized()
{
  auto area = getLocalBounds().reduced(4);
  
  AIthinkingButton->setVisible(false);
  AIreadyButton->setVisible(false);
  AIthinkingButton->setBounds(area);
  AIreadyButton->setBounds(area);


}

void AIOverlay::paint(juce::Graphics &g) {
  if(parent->sCam && parent->sCam->classifying && AIready) {
    AIreadyButton->setVisible(true);
  }
    /*
    bool compositing = parent->sCam->compositing;
    bool done = parent->sCam->tileEmbeddingComplete;

    if(compositing && !done){
      AIthinkingButton->setVisible(true);
    }

    if(compositing && done){
      AIreadyButton->setVisible(true);
    }
  }

  if(AIthinkingButton->isVisible()){
    setAlpha(currentOpacity);
  }
  */
  Component::paint(g);
}


void AIOverlay::buttonClicked(juce::Button* button)
{
  if (button == AIreadyButton.get())
  {
    if(sCam){
      if(t.joinable()){
        t.join();
      }
      t = std::thread([strmCam = sCam](){
        strmCam->run_agg_classify();
      });
    }
    parent->parent->capture->shadeClasses = true;
  }
  resized();
}


void AIOverlay::timerCallback()
{
  // Update the opacity value to create the fade animation
  const float speed = 0.2f; // Change rate per frame

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

//void AIOverlay::set_sCam(std::shared_ptr<pathCam::StreamCam> sCam) {
//  aic_r = new AIClassify(sCam);
//}


