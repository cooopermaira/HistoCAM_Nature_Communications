//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

class bcamThread final : public juce::Thread {
public:
    explicit bcamThread(const juce::String &threadName, MainComponent *parent,
                        std::shared_ptr<pathCam::StreamCam> bcam) : Thread(threadName), parent(parent),
                                                                    bcam(bcam) {};
    void run() override{
        bcam->run();

    }

    MainComponent *parent;
    std::shared_ptr<pathCam::StreamCam> bcam;
};

class bcamPocoRunnable: public Poco::Runnable{
public:
    bcamPocoRunnable(std::shared_ptr<pathCam::StreamCam> bcam):bcam(bcam){};
    std::shared_ptr<pathCam::StreamCam> bcam;

    virtual void run(){
        bcam->run();
    }

};


void CaptureComponent::startRecording() {
    recording = true;

    //bcam->run();
    parent->imageview->setImage(parent->MRimage);
    parent->capture->setImage(parent->MRimage);
    parent->annotate->setImage(parent->MRimage);



    auto bcamRunnable = new bcamPocoRunnable(bcam);
    bcamThread.start(bcamRunnable);

    //(new bcamThread("bcam Thread", parent, bcam))->run();
    repaint();
}

