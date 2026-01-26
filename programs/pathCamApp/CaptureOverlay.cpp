//
//  CaptureOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

CaptureOverlay::CaptureOverlay(CaptureComponent *parent,
                 StringArray &iconNames,
                 OwnedArray<Drawable> &iconsFromZipFile) : parent(parent) {

  for (int i = 0; i < iconNames.size(); i++) {
    if (iconNames[i] == "record.svg") {
      recordButton.reset(new SvgButton("record", iconsFromZipFile[i]));
      recordButton->addListener(this);
      addAndMakeVisible(*recordButton);
    }

    if (iconNames[i] == "stop.svg") {
      stopButton.reset(new SvgButton("stop", iconsFromZipFile[i]));
      stopButton->addListener(this);
      addAndMakeVisible(*stopButton);
    }

    if (iconNames[i] == "simulate.svg") {
      simulateButton.reset(new SvgButton("simulate", iconsFromZipFile[i]));
      simulateButton->addListener(this);
      addAndMakeVisible(*simulateButton);
    }

    if (parent->controlsOverlay->slideListButton) {
      parent->controlsOverlay->slideListButton->addListener(this);
    }

  }
}
void CaptureOverlay::resized()
{
  auto area = getLocalBounds().reduced (4);
  if(parent->simulating || parent->recording){
    simulateButton->setVisible(false);
    recordButton->setVisible(false);
    stopButton->setVisible(true);
    stopButton->setBounds(area.removeFromRight(100).reduced(20,0));
    parent->controlsOverlay->slideListButton->setVisible(false);
  }else{
    simulateButton->setVisible(true);
    if (parent->sCam) {
      parent->controlsOverlay->slideListButton->setVisible(true);
    }else {
      parent->controlsOverlay->slideListButton->setVisible(false);
    }
#ifdef WITH_SPINNAKER
    recordButton->setVisible(true);
#else
    recordButton->setVisible(false);
#endif
    stopButton->setVisible(false);
    simulateButton->setBounds(area.removeFromRight(100).reduced(20,0));
    recordButton->setBounds(area.removeFromRight(100).reduced(20,0));
  }
}


void CaptureOverlay::buttonClicked(juce::Button* button)
{
  if (button == parent->controlsOverlay->slideListButton.get()) {
    parent->labelList->refresh();
    parent->labelList->setVisible(!parent->labelList->isVisible());
  }
  if (button == recordButton.get())
  {
    parent->startRecording();
  }
  if (button == simulateButton.get()){
    parent->startSimulating();
  }
  if (button == stopButton.get())
  {
    parent->stop();
  }

  resized();
  parent->resized();

}
