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

std::pair<std::string, std::vector<tsWord> > send_transcribe_call(juce::File audioFile) {
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
          std::string fullText;

          if (jsonObj->hasProperty("text")) {
            fullText = jsonObj->getProperty("text").toString().toStdString();
          }

          juce::var wordsVar = jsonObj->getProperty("words");
          auto *wordsArr = wordsVar.getArray();
          if (!wordsArr) {
            return {};
          }

          ans.reserve((size_t) wordsArr->size());
          for (const auto &wv: *wordsArr) {
            auto *wobj = wv.getDynamicObject();
            if (wobj == nullptr) { continue; }

            tsWord w;
            w.word = wobj->getProperty("word").toString().toStdString();
            w.startMS = (int) wobj->getProperty("start_ms");
            w.endMS = (int) wobj->getProperty("end_ms");

            if (!w.word.empty())
              ans.push_back(std::move(w));
          }
          return {fullText, ans};
        }
      }
    }
  }
  return {};
}

static std::string buildResponsesRequestBody_JSON(const juce::String &text) {
  const juce::String systemMsg =
      "You convert pathology slide-review transcripts into a sequence of short annotation labels.\n"
      "Input: a single string 'text'.\n"
      "Define the transcript word list as: split 'text' on single spaces. Indices refer to this list (0-based).\n"
      "Rules:\n"
      "1) Output ONLY JSON matching the provided schema.\n"
      "2) Produce 1..N annotations in the SAME ORDER the ideas appear in the transcript.\n"
      "3) Each annotation corresponds to a contiguous span of the word list: span_start_i..span_end_i (inclusive).\n"
      "4) Do NOT merge non-contiguous mentions: if topic A then B then A again, output three annotations.\n"
      "5) label must be 1-5 words.\n"
      "6) label is a semantic reduction / canonical phrase for the span. It DOES NOT need to be an exact substring.\n"
      "   Example: span contains 'Gleason pattern 3+4' -> label 'Gleason 3+4'.\n"
      "7) Prefer canonical pathology wording (e.g., 'perineural invasion', 'positive margin', 'Gleason 3+4').\n"
      "8) Choose spans that tightly cover the evidence words for the idea (include necessary modifiers like 3+4).\n";

  // Helper to build {"type": "..."} objects for schema leaf nodes
  auto makeTypeObj = [](const juce::String &t) -> juce::var {
    juce::DynamicObject::Ptr o(new juce::DynamicObject());
    o->setProperty("type", t);
    return {o.get()};
  };

  // --- Build item schema: {label, span_start_i, span_end_i}
  juce::DynamicObject::Ptr annProps(new juce::DynamicObject());
  annProps->setProperty("label", makeTypeObj("string"));
  annProps->setProperty("span_start_i", makeTypeObj("integer"));
  annProps->setProperty("span_end_i", makeTypeObj("integer"));

  // required: ["label","span_start_i","span_end_i"]  (build array safely)
  juce::Array<juce::var> annRequiredArr;
  annRequiredArr.add("label");
  annRequiredArr.add("span_start_i");
  annRequiredArr.add("span_end_i");
  juce::var annRequiredVar = annRequiredArr;

  juce::DynamicObject::Ptr annItem(new juce::DynamicObject());
  annItem->setProperty("type", "object");
  annItem->setProperty("properties", juce::var(annProps.get()));
  annItem->setProperty("required", annRequiredVar);
  annItem->setProperty("additionalProperties", false);

  juce::DynamicObject::Ptr annotationsProp(new juce::DynamicObject());
  annotationsProp->setProperty("type", "array");
  annotationsProp->setProperty("items", juce::var(annItem.get()));

  // --- Root schema: { annotations: [...] }
  juce::DynamicObject::Ptr rootProps(new juce::DynamicObject());
  rootProps->setProperty("annotations", juce::var(annotationsProp.get()));

  juce::Array<juce::var> rootRequiredArr;
  rootRequiredArr.add("annotations");
  juce::var rootRequiredVar = rootRequiredArr;

  juce::DynamicObject::Ptr schemaObj(new juce::DynamicObject());
  schemaObj->setProperty("type", "object");
  schemaObj->setProperty("properties", juce::var(rootProps.get()));
  schemaObj->setProperty("required", rootRequiredVar);
  schemaObj->setProperty("additionalProperties", false);

  // --- Responses: text.format
  juce::DynamicObject::Ptr formatObj(new juce::DynamicObject());
  formatObj->setProperty("type", "json_schema");
  formatObj->setProperty("name", "annotation_extraction");
  formatObj->setProperty("strict", true);
  formatObj->setProperty("schema", juce::var(schemaObj.get()));

  juce::DynamicObject::Ptr textObj(new juce::DynamicObject());
  textObj->setProperty("format", juce::var(formatObj.get()));

  // --- Root request body
  juce::DynamicObject::Ptr root(new juce::DynamicObject());
  root->setProperty("model", "gpt-4o-mini");
  root->setProperty("instructions", systemMsg);
  root->setProperty("input", text);
  root->setProperty("temperature", 0);
  root->setProperty("text", juce::var(textObj.get()));

  return juce::JSON::toString(juce::var(root.get()), true).toStdString();
}


static std::string openAIResponses_POST(const std::string &apiKey,
                                        const std::string &requestBodyJson,
                                        int timeoutMs = 60000) {
  juce::URL url("https://api.openai.com/v1/responses");

  juce::String headers;
  headers << "Authorization: Bearer " << apiKey << "\r\n";
  headers << "Content-Type: application/json\r\n";

  auto in = url.withPOSTData(requestBodyJson)
      .createInputStream(
        juce::URL::InputStreamOptions(juce::URL::ParameterHandling::inPostData)
        .withExtraHeaders(headers)
        .withConnectionTimeoutMs(timeoutMs)
        .withNumRedirectsToFollow(0)
      );

  if (in == nullptr) {
    return {};
  }

  return in->readEntireStreamAsString().toStdString();
}

static std::vector<AnnotationSpan> parseAnnotationsFromResponses(const std::string &responsesJson) {
  std::vector<AnnotationSpan> out;

  auto top = juce::JSON::parse(responsesJson);
  if (!top.isObject()) { return out; }

  auto *topObj = top.getDynamicObject();
  if (!topObj) { return out; }

  auto outputVar = topObj->getProperty("output");
  auto *outArr = outputVar.getArray();
  if (!outArr) { return out; }

  // 1) Find the first message
  juce::var messageContentVar; // this will become the "content" array inside the message
  for (const auto &item: *outArr) {
    auto *msgObj = item.getDynamicObject();
    if (!msgObj) { continue; }

    if (msgObj->getProperty("type").toString() == "message") {
      messageContentVar = msgObj->getProperty("content");
      break;
    }
  }

  auto *msgContentArr = messageContentVar.getArray();
  if (!msgContentArr) { return out; }

  // 2) Find output_text (or output_json if it ever appears)
  juce::String payloadText;

  for (const auto &c: *msgContentArr) {
    auto *cobj = c.getDynamicObject();
    if (!cobj) { continue; }

    const auto ctype = cobj->getProperty("type").toString();

    if (ctype == "output_text") {
      // In your response, this is the JSON string
      payloadText = cobj->getProperty("text").toString();
      break;
    } else if (ctype == "output_json") {
      // Some responses may return JSON directly; handle it too.
      auto jsonVar = cobj->getProperty("content");
      auto *jsonObj = jsonVar.getDynamicObject();
      if (jsonObj) {
        // Parse annotations directly from jsonObj
        auto annVar = jsonObj->getProperty("annotations");
        auto *annArr = annVar.getArray();
        if (!annArr) {
          return out;
        }

        out.reserve((size_t) annArr->size());
        for (const auto &av: *annArr) {
          auto *aobj = av.getDynamicObject();
          if (!aobj) { continue; }

          AnnotationSpan a;
          a.label = aobj->getProperty("label").toString().toStdString();
          a.spanStartI = (int) aobj->getProperty("span_start_i");
          a.spanEndI = (int) aobj->getProperty("span_end_i");
          if (!a.label.empty()) {
            out.push_back(std::move(a));
          }
        }
        return out;
      }
    }
  }

  if (payloadText.isEmpty()) {
    return out;
  }

  // 3) payloadText is a JSON string like: {"annotations":[...]}
  auto inner = juce::JSON::parse(payloadText);
  if (!inner.isObject()) {
    return out;
  }

  auto *innerObj = inner.getDynamicObject();
  if (!innerObj) {
    return out;
  }

  auto annVar = innerObj->getProperty("annotations");
  auto *annArr = annVar.getArray();
  if (!annArr) {
    return out;
  }

  out.reserve((size_t) annArr->size());

  for (const auto &av: *annArr) {
    auto *aobj = av.getDynamicObject();
    if (!aobj) { continue; }

    AnnotationSpan a;
    a.label = aobj->getProperty("label").toString().toStdString();
    a.spanStartI = (int) aobj->getProperty("span_start_i");
    a.spanEndI = (int) aobj->getProperty("span_end_i");

    if (!a.label.empty()) {
      out.push_back(std::move(a));
    }
  }

  return out;
}


std::vector<AnnotationSpan> reduceToAnnotations_OpenAI(const std::string &text) {
  std::string apiKey(
    "REDACTED_OPENAI_API_KEY");

  const std::string body = buildResponsesRequestBody_JSON(text);

  const std::string resp = openAIResponses_POST(apiKey, body, 60000);
  if (resp.empty()) {
    return {};
  }

  return parseAnnotationsFromResponses(resp);
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
    auto start = std::chrono::high_resolution_clock::now();
    auto [fullText,wordVec] = send_transcribe_call(dictPath);
    auto annoSpanVec = reduceToAnnotations_OpenAI(fullText);
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout<<"full text: \""<<fullText<<"\" processed in "<<dur<<std::endl;
    std::shared_ptr<MRTiledImageSet> mrImgSet;
    {
      Poco::FastMutex::ScopedLock lock(parent->sCam->previousSlidesMutex);
      mrImgSet = parent->sCam->previousSlides[index];
    }

    if (annoSpanVec.empty()) {
      continue;
    }

    allSlideAnnotationMutex.lock();
    auto thisSlidesAnnotations = allSlideAnnotations[index];
    allSlideAnnotationMutex.unlock();

    //make the polygon
    for (auto &annospan: annoSpanVec) {
      if (annospan.spanStartI < 0 || annospan.spanEndI < 0) {
        std::cout << "bad text indices for annotation: " + annospan.label << std::endl;
        continue;
      }

      annospan.endMS = wordVec[annospan.spanEndI].endMS;
      annospan.startMS = wordVec[annospan.spanStartI].startMS;

      long startFrameIndex, endFrameIndex;
      auto polyAnnoVertices = mrImgSet->poly_annotation_from_time_interval(annospan.startMS,
                                                                           annospan.endMS,
                                                                           startFrameIndex,
                                                                           endFrameIndex);

      annospan.startFrameIdx = startFrameIndex;
      annospan.endFrameIdx = endFrameIndex;

      // Create a new polygon annotation with the label
      auto polyAnno = std::make_shared<VoicePointPoly>(annospan);
      std::cout<<"annospan frame index: "<<annospan.startFrameIdx<<" "<<annospan.endFrameIdx<<std::endl;

      // Add each vertex from polyAnnoVertices
      for (const auto &vertex: polyAnnoVertices) {
        polyAnno->add(fPoint(vertex.x, vertex.y));
      }

      // Add the polygon annotation to this slide's annotations
      thisSlidesAnnotations->push_back(polyAnno);
    }

    // Update the list box if this is the current slide
    if (thisSlidesAnnotations == activeAnnotations) {
      leftComponent->requestListRefresh();
      MessageManager::callAsync([this]{rightComponent->repaint();});
    }
  }

  newVoiceAnnotation.wait();
}

void AnnotateComponent::silly_test() {
  juce::File dictPath("/home/cm/Documents/data/blur_test/config/0/dictation.wav");
  auto [fullText,wordVec] = send_transcribe_call(dictPath);
  auto annoSpanVec = reduceToAnnotations_OpenAI(fullText);
  int k = 0;
}
