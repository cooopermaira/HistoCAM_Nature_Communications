//
//  CaptureComponent.hpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef CaptureComponent_h
#define CaptureComponent_h

#include "JuceHeader.h"

#ifdef WITH_SPINNAKER
class SpinPath;
#else
class StreamCam;
#endif


class CaptureComponent : public ImageViewComponent {
  
  friend class CaptureOverlay;
  
public:
    CaptureComponent(std::shared_ptr<fRectangle> view,
        StringArray& iconNames,
        OwnedArray<Drawable>& iconsFromZipFile, Poco::Util::LayeredConfiguration::Ptr config,
        MainComponent* parent);
    
  Poco::Util::LayeredConfiguration::Ptr config;

    
#ifdef WITH_SPINNAKER
    std::shared_ptr<pathCam::SpinPath> bcam;
#endif

    std::shared_ptr<pathCam::StreamCam> sCam;

    MainComponent *parent;

    void resized() {

        ImageViewComponent::resized();

        {
            const ScopedLock lock(mutex);
            juce::Rectangle<int> b = getLocalBounds();
            int width = 300;
            captureOverlay->setBounds(juce::Rectangle<int>(b.getWidth() - width - 20, 20, width, 60));

        }

    }

    bool keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) {
        ImageViewComponent::keyPressed(key, originatingComponent);

        if (!isVisible()) { return false; }

        if (key.getKeyCode() == KeyPress::spaceKey) {
          if (recording || simulating) { stop(); }
        }
        return false;  // Key press not handled
    }

    void startRecording();
  
    void startSimulating();

    void stop();
  
    void stopRecording();
  
    void stopSimulating();

    void drawSlide(juce::Graphics& g, float scale) override{
      ImageViewComponent::drawSlide(g, scale);
      
      if(view->isEmpty() || MRImage->empty()){ return; }
      
      cv::Rect_<float> frameBox;
      bool showAsCircle;
      sCam->get_last_frame(frameBox, showAsCircle);

      auto bounds = RectCtoJ < float >(frameBox);

      bounds *= view2screenScale(*view) * scale;
      
      g.setColour(juce::Colours::red);
//      if(false){
        auto center = bounds.getCentre();
        fPoint radius = sCam->get_scope_radius() * view2screenScale(*view) * scale;;
        center -= radius/2.0;
        g.drawEllipse (center.getX(), center.getY(), radius.getX(), radius.getY(), 3);

//      }else{
        g.drawRect(bounds, 3);
//      }

    }


    void paint(juce::Graphics &g) override {
        ImageViewComponent::paint(g);

        if (recording) {
            g.setColour(juce::Colours::red);
            g.drawRect(getLocalBounds(), 3);
            std::string r = "Recording...";
            g.setFont(30);
            auto b = getLocalBounds().removeFromRight(175).removeFromBottom(50);
            g.drawText(r, b.getX(), b.getY(), 200, 40, Justification::left);
        }


    }

private:
    std::unique_ptr<CaptureOverlay> captureOverlay;
    Poco::Thread compositeThread;
    bool recording;
    bool simulating;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CaptureComponent)
};

#endif /* CaptureComponent_h */

