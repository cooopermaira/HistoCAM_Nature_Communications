//
//  CaptureComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"


class bcamPocoRunnable: public Poco::Runnable{
public:
    bcamPocoRunnable(CaptureComponent* cptcmp):cptcmp(cptcmp) {};
    CaptureComponent* cptcmp;
    virtual void run(){
        cptcmp->bcam->run();
    }

};

CaptureComponent::CaptureComponent(std::shared_ptr<fRectangle> view,
    StringArray& iconNames,
    OwnedArray<Drawable>& iconsFromZipFile, Poco::Util::LayeredConfiguration::Ptr config,
    MainComponent* parent) : config(config), parent(parent),
ImageViewComponent(view, iconNames, iconsFromZipFile), recording(false) {
    
#ifdef WITH_SPINNAKER
    bcam.reset(new pathCam::SpinPath(config));
#else
    bcam.reset(new pathCam::StreamCam(config));
#endif
  
    bcam->add_observer(parent);
    MRImage = bcam->get_image_reference();
    captureOverlay.reset(new CaptureOverlay(this, iconNames, iconsFromZipFile));
    addAndMakeVisible(captureOverlay.get());


}
void CaptureComponent::startRecording() {
    recording = true;

    //bcam->run();
    parent->imageview->setImage(parent->MRimage);
    parent->capture->setImage(parent->MRimage);
    parent->annotate->setImage(parent->MRimage);


    auto bcamRunnable = new bcamPocoRunnable(this);
    bcamThread.start(bcamRunnable);

    //(new bcamThread("bcam Thread", parent, bcam))->run();
    repaint();
}

