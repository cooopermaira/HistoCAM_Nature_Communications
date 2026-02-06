//
//  AnnotateComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

void AnnotateComponent::removeSelected() {
  if (!selected)
    return;

  // Remove annotation object from the annotations list
  auto itAnno = std::find(activeAnnotations->begin(), activeAnnotations->end(), selected);
  if (itAnno == activeAnnotations->end())
    return;

  if (selected->getType() == Annotation::_SEG) {
    // Safe cast (never dereference a failed dynamic_cast)
    auto seg = std::dynamic_pointer_cast<SegmentAnnotation>(selected);
    if (seg) {
      const int targetSegID = seg->ID;

      auto *mr0 = rightComponent->MRImageSet->MRImages[0].get();
      for (auto &tileIdx: mr0->liveTiles) {
        auto tileObj = mr0->get_base_tile(tileIdx);

        auto &masks = tileObj->SAMMasks; // std::map<int, std::pair<cv::cuda::GpuMat, void*>>

        auto itMask = masks.find(targetSegID);
        if (itMask != masks.end()) {
          // Release GPU memory
          itMask->second.first.release();

          // Delete frontend cached object, then null it
          if (itMask->second.second) {
            delete static_cast<juce::Image *>(itMask->second.second);
            itMask->second.second = nullptr;
          }

          // Remove the entry from the map
          masks.erase(itMask);
        }
      }
    }
  }

  activeAnnotations->erase(itAnno);
  selected.reset();

  leftComponent->updatelist();
  repaint();
}


void AnnotateComponent::setImage(std::shared_ptr<MRTiledImageSet> image) {
  rightComponent->setImage(image);
  if (image) { update_active_annotations(image->index); }
}

std::vector<tsWord> send_transcribe_call(juce::File audioFile) {
  // Send audio file to local transcription server
  juce::URL transcriptionUrl("http://127.0.0.1:8088/v1/audio/transcriptions");

  // Read the audio file
  juce::MemoryBlock audioData;
  if (audioFile.loadFileAsData(audioData)) {
    // Build multipart form data body
    juce::MemoryOutputStream bodyStream;
    juce::String boundary = "----WebKitFormBoundary" + juce::String::toHexString(
                              juce::Random::getSystemRandom().nextInt());

    // Add file field
    bodyStream << "--" << boundary << "\r\n";
    bodyStream << "Content-Disposition: form-data; name=\"file\"; filename=\"" << audioFile.getFileName() << "\"\r\n";
    bodyStream << "Content-Type: audio/wav\r\n\r\n";
    bodyStream.write(audioData.getData(), audioData.getSize());
    bodyStream << "\r\n";

    // Add language field
    bodyStream << "--" << boundary << "\r\n";
    bodyStream << "Content-Disposition: form-data; name=\"language\"\r\n\r\n";
    bodyStream << "en\r\n";

    // Add chunk_s field
    bodyStream << "--" << boundary << "\r\n";
    bodyStream << "Content-Disposition: form-data; name=\"chunk_s\"\r\n\r\n";
    bodyStream << "30\r\n";

    // Add overlap_s field
    bodyStream << "--" << boundary << "\r\n";
    bodyStream << "Content-Disposition: form-data; name=\"overlap_s\"\r\n\r\n";
    bodyStream << "2\r\n";

    // Add word_timestamps field
    bodyStream << "--" << boundary << "\r\n";
    bodyStream << "Content-Disposition: form-data; name=\"word_timestamps\"\r\n\r\n";
    bodyStream << "true\r\n";

    // End boundary
    bodyStream << "--" << boundary << "--\r\n";

    // Create a POST request with the body data
    juce::String extraHeaders = "Content-Type: multipart/form-data; boundary=" + boundary;

    auto inputStream = transcriptionUrl.withPOSTData(bodyStream.getMemoryBlock())
        .createInputStream(
          juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inPostData)
          .withExtraHeaders(extraHeaders)
          .withConnectionTimeoutMs(30000)
          .withNumRedirectsToFollow(0)
        );

    if (inputStream != nullptr) {
      // Read the response
      juce::String response = inputStream->readEntireStreamAsString();

      // Parse JSON response
      auto jsonResult = juce::JSON::parse(response);
      if (jsonResult.isObject()) {
        // Process the transcription result
        auto jsonObj = jsonResult.getDynamicObject();
        if (jsonObj != nullptr) {
          std::vector<tsWord> ans;

          juce::var wordsVar = jsonObj->getProperty("words");
          auto *wordsArr = wordsVar.getArray();
          if (!wordsArr) {
            return {};
          }

          ans.reserve((size_t) wordsArr->size());
          for (const auto &wv: *wordsArr) {
            auto *wobj = wv.getDynamicObject();
            if (wobj == nullptr){continue;}

            tsWord w;
            w.word = wobj->getProperty("word").toString().toStdString();
            w.startMS = (int) wobj->getProperty("start_ms");
            w.endMS = (int) wobj->getProperty("end_ms");

            if (!w.word.empty())
              ans.push_back(std::move(w));
          }
          return ans;
        }
      }
    }
  }
  return {};
}

void AnnotateComponent::voice_annotation_handler() {
  while (voiceHandlerShouldContinue) {
    int index = -1;
    juce::File dictPath;
    {
      Poco::FastMutex::ScopedLock lock(voiceAnnoMutex);
      if (!voiceAnnoOutstanding.empty()) {
        auto [i,p] = voiceAnnoOutstanding.front();
        voiceAnnoOutstanding.pop();
        index = i;
        dictPath = p;
      } else {
        newVoiceAnnotation.reset();
      }
    }
    if (index < 0) {
      newVoiceAnnotation.wait();
      continue;
    }
    auto ans = send_transcribe_call(dictPath);
    int k = 0;
  }

  newVoiceAnnotation.wait();
}
