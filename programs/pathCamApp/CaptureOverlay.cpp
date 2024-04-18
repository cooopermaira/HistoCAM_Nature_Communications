//
//  CaptureOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"


void CaptureOverlay::buttonClicked(juce::Button* button)
{
  if (button == recordButton.get())
  {
    parent->startRecording();
  }
  if (button == stopButton.get())
  {
    parent->stopRecording();
  }
}
