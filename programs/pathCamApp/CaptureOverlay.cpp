//
//  CaptureOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"


void CaptureOverlay::resized()
{
  auto area = getLocalBounds().reduced (4);
  if(parent->simulating || parent->recording){
    simulateButton->setVisible(false);
    recordButton->setVisible(false);
    stopButton->setVisible(true);
    stopButton->setBounds(area.removeFromRight(100).reduced(20,0));
  }else{
    simulateButton->setVisible(true);
    recordButton->setVisible(true);
    stopButton->setVisible(false);
    simulateButton->setBounds(area.removeFromRight(100).reduced(20,0));
    recordButton->setBounds(area.removeFromRight(100).reduced(20,0));
  }
}


void CaptureOverlay::buttonClicked(juce::Button* button)
{
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
}
