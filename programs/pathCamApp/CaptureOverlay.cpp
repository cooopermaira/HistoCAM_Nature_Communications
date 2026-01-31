//
//  CaptureOverlay.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

bool StreamCamLabelList::slideHasMatchingAnnotation(int slideIndex, const juce::String &searchText) {
  // Check if the slide label matches the search text
  if (slideIndex >= 0 && slideIndex < sCam->get_num_slides()) {
    juce::String slideLabel = sCam->get_slide_label(slideIndex);
    if (slideLabel.containsIgnoreCase(searchText)) {
      return true;
    }
  }

  // Check if any annotation in the slide matches the search text
  return slideHasAnnotationMatch(slideIndex, searchText);
}

bool StreamCamLabelList::slideHasAnnotationMatch(int slideIndex, const juce::String &searchText) {
  // Check if any annotation in the slide matches the search text
  if (!annotateComp)
    return false;

  // Check if the slide index is valid in allSlideAnnotations
  if (slideIndex < 0 || slideIndex >= (int) annotateComp->allSlideAnnotations.size())
    return false;

  auto slideAnnotations = annotateComp->allSlideAnnotations[slideIndex];
  if (!slideAnnotations)
    return false;

  // Search through all annotations in this slide
  for (const auto &anno: *slideAnnotations) {
    if (anno && anno->getName().containsIgnoreCase(searchText)) {
      return true;
    }
  }

  return false;
}


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
  }
}

void CaptureOverlay::resized() {
  auto area = getLocalBounds().reduced(4);
  if (parent->simulating || parent->recording) {
    simulateButton->setVisible(false);
    recordButton->setVisible(false);
    stopButton->setVisible(true);
    stopButton->setBounds(area.removeFromRight(100).reduced(20, 0));
  } else {
    simulateButton->setVisible(true);
#ifdef WITH_SPINNAKER
    recordButton->setVisible(true);
#else
    recordButton->setVisible(false);
#endif
    stopButton->setVisible(false);
    simulateButton->setBounds(area.removeFromRight(100).reduced(20, 0));
    recordButton->setBounds(area.removeFromRight(100).reduced(20, 0));
  }

  // Trigger MainComponent to update slideListButton and labelList visibility
  parent->parent->resized();
}


void CaptureOverlay::buttonClicked(juce::Button *button) {
  if (button == recordButton.get()) {
    parent->startRecording();
  }
  if (button == simulateButton.get()) {
    parent->startSimulating();
  }
  if (button == stopButton.get()) {
    parent->stop();
  }

  resized();
  parent->resized();
}
