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
/*
class WavRecorder : private juce::AudioIODeviceCallback
{
public:
    bool initialised = false;
    juce::File writeFile;

    // Device runs at whatever it wants; we always WRITE 16 kHz mono.
    double deviceSampleRate = 48000.0;
    static constexpr double targetSampleRate = 16000.0;

    juce::LagrangeInterpolator resampler;

    WavRecorder() : writerThread("WavRecorderThread")
    {
        writerThread.startThread();
    }

    ~WavRecorder() override
    {
        stop();
        if (initialised)
            deviceManager.removeAudioCallback(this);
    }

    // Call once.
    void init(int numInputChannels = 1)
    {
        // Do NOT force sample rate here; let the device open fast.
        deviceManager.initialise(numInputChannels, 0, nullptr, true);
        deviceManager.addAudioCallback(this);
        initialised = true;
    }

    bool startRecording(const juce::File& fileToWrite)
    {
        stop(); // stop existing recording if any

        writeFile = fileToWrite;
        fileToWrite.deleteFile();

        auto stream = std::unique_ptr<juce::FileOutputStream>(fileToWrite.createOutputStream());
        if (!stream)
            return false;

        auto* device = deviceManager.getCurrentAudioDevice();
        if (!device)
            return false;

        // IMPORTANT: Writer must match the data we write (16 kHz, mono).
        const double writerSampleRate = targetSampleRate;
        const int outChans = 1;

        juce::WavAudioFormat wav;
        std::unique_ptr<juce::AudioFormatWriter> w(
            wav.createWriterFor(stream.get(),
                                writerSampleRate,
                                (unsigned int) outChans,
                                16,
                                {},
                                0));

        if (!w)
            return false;

        stream.release(); // writer now owns the stream

        // Queue size in samples; 32768 is fine. You can increase if you want.
        threadedWriter.reset(new juce::AudioFormatWriter::ThreadedWriter(
            w.release(), writerThread, 32768));

        {
            const juce::ScopedLock sl(writerLock);
            activeWriter = threadedWriter.get();
        }

        recordingStart = juce::Time::getMillisecondCounterHiRes();
        return true;
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

    double getRecordingMs() const
    {
        return juce::Time::getMillisecondCounterHiRes() - recordingStart;
    }

private:
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override
    {
        deviceSampleRate = device ? device->getCurrentSampleRate() : 48000.0;
        DBG("Audio device: " << (device ? device->getName() : "null"));
        DBG("Device SR: " << deviceSampleRate << "  Buffer: " << (device ? device->getCurrentBufferSizeSamples() : -1));
        resampler.reset();

        const int expectedIn = device ? device->getCurrentBufferSizeSamples() : 1024;
        ensureScratchSizes(expectedIn);
        outPtrArray[0] = monoOutScratch.getReadPointer(0);
    }


    void audioDeviceStopped() override {}

    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
                                      int numInputChannels,
                                      float* const* outputChannelData,
                                      int numOutputChannels,
                                      int numSamples,
                                      const juce::AudioIODeviceCallbackContext& context) override
{
    if (numSamples <= 0)
        return;

    // Grab writer once; avoid holding lock during DSP
    juce::AudioFormatWriter::ThreadedWriter* writer = nullptr;
    {
        const juce::ScopedLock sl(writerLock);
        writer = activeWriter;
    }
    if (writer == nullptr)
        return;

    ensureScratchSizes(numSamples);

    // 1) Downmix to mono into monoInScratch (no allocation)
    monoInScratch.clear(0, 0, numSamples);

    if (numInputChannels > 0)
    {
        const float scale = 1.0f / (float) numInputChannels;

        for (int ch = 0; ch < numInputChannels; ++ch)
        {
            if (const float* in = inputChannelData[ch])
                monoInScratch.addFrom(0, 0, in, numSamples, scale);
        }
    }

    const double inRate  = (deviceSampleRate > 0.0) ? deviceSampleRate : 48000.0;
    const double outRate = targetSampleRate;

    // If device already runs at 16k, write directly
    if (std::abs(inRate - outRate) < 1e-6)
    {
        const float* chans[] = { monoInScratch.getReadPointer(0) };
        writer->write(chans, numSamples);
        return;
    }

    // 2) Resample monoInScratch -> monoOutScratch at 16k
    // LagrangeInterpolator expects ratio = inputSamplesPerOutputSample
    const double ratio = inRate / outRate;

    // CRITICAL FIX:
    // Only ask for the number of output samples we can safely produce from numSamples input,
    // otherwise the resampler will read past the end of monoInScratch and you'll get clicks/garbage.
    int outSamplesDesired = (int) std::floor((double) numSamples / ratio);
    if (outSamplesDesired <= 0)
        return;

    // Ensure output scratch can hold exactly what we'll generate
    if (monoOutScratch.getNumSamples() < outSamplesDesired)
    {
        monoOutScratch.setSize(1, outSamplesDesired, false, false, true);
        outPtrArray[0] = monoOutScratch.getReadPointer(0);
    }

    float* out = monoOutScratch.getWritePointer(0);
    const float* in = monoInScratch.getReadPointer(0);

    const int outSamples = resampler.process(ratio, in, out, outSamplesDesired);
    if (outSamples <= 0)
        return;

    // 3) Write resampled audio (16k) - matches WAV header created in startRecording()
    writer->write(outPtrArray.data(), outSamples);
}


    // Ensures scratch buffers exist and are large enough for current callback size.
    // Only grows; avoids repeated allocations.
    void ensureScratchSizes(int inSamples)
    {
        if (inSamples <= 0)
            return;

        // Input scratch (mono at device rate)
        if (monoInScratch.getNumSamples() < inSamples)
            monoInScratch.setSize(1, inSamples, false, false, true);

        // Output scratch (mono at 16k) sized for *safe* max output for this block.
        // ratio = inputSamplesPerOutputSample
        const double inRate = (deviceSampleRate > 0.0) ? deviceSampleRate : 48000.0;
        const double ratio  = inRate / targetSampleRate;

        // We will only ever ask the resampler to produce floor(inSamples / ratio) samples,
        // so size output scratch to at least that.
        int outSamplesDesired = (int) std::floor((double) inSamples / ratio);
        outSamplesDesired = juce::jmax(1, outSamplesDesired);

        if (monoOutScratch.getNumSamples() < outSamplesDesired)
        {
            monoOutScratch.setSize(1, outSamplesDesired, false, false, true);
            outPtrArray[0] = monoOutScratch.getReadPointer(0);
        }
    }


    juce::AudioDeviceManager deviceManager;

    juce::TimeSliceThread writerThread;
    std::unique_ptr<juce::AudioFormatWriter::ThreadedWriter> threadedWriter;

    juce::CriticalSection writerLock;
    juce::AudioFormatWriter::ThreadedWriter* activeWriter = nullptr;

    // Scratch buffers to avoid per-callback allocations
    juce::AudioBuffer<float> monoInScratch  { 1, 0 };
    juce::AudioBuffer<float> monoOutScratch { 1, 0 };

    // Stable pointer array for ThreadedWriter::write
    std::array<const float*, 1> outPtrArray { { nullptr } };

    double recordingStart = 0.0;
};
*/


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

private:
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
