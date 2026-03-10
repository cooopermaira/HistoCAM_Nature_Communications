//
//  CaptureComponent.hpp
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef CaptureComponent_h
#define CaptureComponent_h

#include "JuceHeader.h"
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_audio_formats/juce_audio_formats.h>



#ifdef WITH_SPINNAKER
class SpinPath;
#else
class StreamCam;
#endif


class WavRecorder : private juce::AudioIODeviceCallback
{
public:

    bool initialised = false;
    juce::File writeFile;

    WavRecorder() : writerThread("WavRecorderThread")
    {
        writerThread.startThread();
    }

    ~WavRecorder() override
    {
        stop();
        deviceManager.removeAudioCallback(this);
    }

    // Call once, after you have permissions, etc.
    void init(int sampleRate = 16000, int numInputChannels = 1)
    {
        juce::AudioDeviceManager::AudioDeviceSetup setup;
        deviceManager.initialise(numInputChannels, 0, nullptr, true);

        // You can optionally force sample rate / buffer sizes here by modifying setup:
        deviceManager.getAudioDeviceSetup(setup);
        setup.sampleRate = sampleRate;
        deviceManager.setAudioDeviceSetup(setup, true);

        deviceManager.addAudioCallback(this);
        initialised = true;
    }

    bool startRecording(const juce::File& fileToWrite)
    {
        stop(); // stop existing recording if any

        writeFile = fileToWrite;
        fileToWrite.deleteFile();
        if (auto stream = std::unique_ptr<juce::FileOutputStream>(fileToWrite.createOutputStream()))
        {
            juce::WavAudioFormat wav;
            auto* device = deviceManager.getCurrentAudioDevice();
            if (!device) return false;

            const double sr = device->getCurrentSampleRate();
            const int inChans = juce::jmax(1, device->getActiveInputChannels().countNumberOfSetBits());

            // We'll write mono even if device provides stereo; downmix in callback.
            const int outChans = 1;

            std::unique_ptr<juce::AudioFormatWriter> w(
                wav.createWriterFor(stream.get(), sr, (unsigned int)outChans, 16, {}, 0));

            if (!w) return false;

            stream.release(); // writer now owns the stream

            threadedWriter.reset(new juce::AudioFormatWriter::ThreadedWriter(
                w.release(), writerThread, 32768));

            {
                const juce::ScopedLock sl(writerLock);
                activeWriter = threadedWriter.get();
            }

            recordingStart = juce::Time::getMillisecondCounterHiRes();
            return true;
        }
        return false;
    }

    void stop()
    {
        {
            const juce::ScopedLock sl(writerLock);
            activeWriter = nullptr;
        }
        threadedWriter.reset();
    }

    bool isRecording() const { return threadedWriter != nullptr; }

    // optional: elapsed time
    double getRecordingMs() const
    {
        return juce::Time::getMillisecondCounterHiRes() - recordingStart;
    }

private:
    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                         int numInputChannels,
                                         float* const* /*outputChannelData*/,
                                         int /*numOutputChannels*/,
                                         int numSamples,
                                         const juce::AudioIODeviceCallbackContext& /*context*/) override
    {
        juce::AudioBuffer<float> monoBuffer(1, numSamples);
        monoBuffer.clear();

        // Downmix: average available input channels
        if (numInputChannels > 0)
        {
            for (int ch = 0; ch < numInputChannels; ++ch)
                if (inputChannelData[ch] != nullptr)
                    monoBuffer.addFrom(0, 0, inputChannelData[ch], numSamples, 1.0f / (float)numInputChannels);
        }

        juce::AudioFormatWriter::ThreadedWriter* writer = nullptr;
        {
            const juce::ScopedLock sl(writerLock);
            writer = activeWriter;
        }

        if (writer != nullptr)
            writer->write(monoBuffer.getArrayOfReadPointers(), numSamples);
    }

    void audioDeviceAboutToStart(juce::AudioIODevice* /*device*/) override {}
    void audioDeviceStopped() override {}

    juce::AudioDeviceManager deviceManager;

    juce::TimeSliceThread writerThread;
    std::unique_ptr<juce::AudioFormatWriter::ThreadedWriter> threadedWriter;

    juce::CriticalSection writerLock;
    juce::AudioFormatWriter::ThreadedWriter* activeWriter = nullptr;

    double recordingStart = 0.0;
};


class CaptureComponent : public ImageViewComponent {
  friend class CaptureOverlay;
  friend class AIOverlay;

public:
  CaptureComponent(std::shared_ptr<fRectangle> view,
                   StringArray &iconNames,
                   OwnedArray<Drawable> &iconsFromZipFile, Poco::Util::LayeredConfiguration::Ptr config,
                   MainComponent *parent);

  Poco::Util::LayeredConfiguration::Ptr config;


  bool recording;
  bool simulating;
  bool ready = false;
  std::string inputPath;
  std::string caseLabel;

#ifdef WITH_SPINNAKER
  std::shared_ptr<pathCam::SpinPath> spinpath;
#endif

  std::shared_ptr<pathCam::StreamCam> sCam;

  MainComponent *parent;

  void set_input(const FileChooser &fc);

  void resized() override{
    auto area = getLocalBounds();

    // IMPORTANT: lay out ImageViewComponent *inside remaining area*
    layoutIn(area);

    {
      const ScopedLock lock(mutex);
      auto b = getLocalBounds();
      int width = 300;
      captureOverlay->setBounds({ b.getWidth() - width - 20, 20, width, 60 });
      aiOverlay->setBounds({ b.getWidth() - 100 - 20,
                             b.getHeight() - 100 - 20, 100, 100 });
    }
  }


  bool keyPressed(const juce::KeyPress &key, juce::Component *originatingComponent) override;

  void startRecording();

  void startSimulating();

  void stop();

  void stopRecording();

  void stopSimulating();

  void drawSlide(juce::Graphics &g, float scale) override;

  void setup_listbox() const;


  void paint(juce::Graphics &g) override {
    ImageViewComponent::paint(g);

    if (ready) {
      g.setColour(Colours::green);
      g.setFont(30);
      auto b = getLocalBounds().removeFromRight(175).removeFromBottom(50);
      g.drawText("Ready", b.getX(), b.getY(), 200, 40, Justification::left);
    }

    if (recording) {
      g.setColour(juce::Colours::red);
      g.drawRect(getLocalBounds(), 3);
      std::string r = "Recording...";
      g.setFont(30);
      auto b = getLocalBounds().removeFromRight(175).removeFromBottom(50);
      g.drawText(r, b.getX(), b.getY(), 200, 40, Justification::left);
    }
  }

    WavRecorder wavRecorder;
  std::unique_ptr<CaptureOverlay> captureOverlay;
  std::unique_ptr<AIOverlay> aiOverlay;
  //std::unique_ptr<ReportOverlay> reportOverlay;

  Poco::Thread compositeThread;
  Poco::Thread updateDrawThread;

  int procedureMode = 0;
  float scopeRadius;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CaptureComponent)
};

#endif /* CaptureComponent_h */
