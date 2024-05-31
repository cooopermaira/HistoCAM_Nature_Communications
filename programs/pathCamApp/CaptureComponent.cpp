//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"
/*
class FileProcessThread final : public juce::Thread{
    explicit FileProcessThread
};
*/
void CaptureComponent::startRecording(){
    recording = true;
    //run streamcam from here
    bcam->run();
    
    parent->imageview->setImage(parent->MRimage);
    parent->capture->setImage(parent->MRimage);
    parent->annotate->setImage(parent->MRimage);

    repaint();
}

