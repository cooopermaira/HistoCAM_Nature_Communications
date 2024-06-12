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
public:
    CaptureComponent(std::shared_ptr<fRectangle> view,
        StringArray& iconNames,
        OwnedArray<Drawable>& iconsFromZipFile, Poco::Util::LayeredConfiguration::Ptr config,
        MainComponent* parent);
    
  Poco::Util::LayeredConfiguration::Ptr config;

    
#ifdef WITH_SPINNAKER
    std::shared_ptr<pathCam::SpinPath> bcam;
#else
    std::shared_ptr<pathCam::StreamCam> bcam;
#endif

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
            if (recording) { stopRecording(); } else { startRecording(); }
        }
        return false;  // Key press not handled
    }

    void startRecording();

    void stopRecording() {
        
#ifdef WITH_SPINNAKER
        bcam->stopCamera();
#endif

        recording = false;
        repaint();
    }

    void paint(juce::Graphics &g) {
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
    Poco::Thread bcamThread;
    bool recording;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (CaptureComponent)
};

#endif /* CaptureComponent_h */
