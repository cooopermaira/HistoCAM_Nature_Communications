//
//  AnnotateComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

namespace {

  std::string apiKey(
    "REDACTED_OPENAI_API_KEY");


  std::string toLower(const std::string &s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return out;
  }

  // ------------------------------------------------------------
  // Helper: normalize string (lowercase + collapse spaces +
  // strip leading/trailing whitespace)
  // ------------------------------------------------------------
  std::string normalize(const std::string &s) {
    std::string out;
    out.reserve(s.size());

    bool prevSpace = false;

    for (unsigned char c: s) {
      if (std::isspace(c)) {
        if (!prevSpace) {
          out.push_back(' ');
          prevSpace = true;
        }
      } else {
        out.push_back(std::tolower(c));
        prevSpace = false;
      }
    }

    // trim
    if (!out.empty() && out.front() == ' ')
      out.erase(out.begin());
    if (!out.empty() && out.back() == ' ')
      out.pop_back();

    return out;
  }

  // ------------------------------------------------------------
  // Split on single spaces (as per your model rule)
  // ------------------------------------------------------------
  std::vector<std::string> splitWords(const std::string &s) {
    std::vector<std::string> words;
    size_t start = 0;

    while (start < s.size()) {
      size_t end = s.find(' ', start);
      if (end == std::string::npos) {
        words.push_back(s.substr(start));
        break;
      }
      words.push_back(s.substr(start, end - start));
      start = end + 1;
    }

    return words;
  }

  // ------------------------------------------------------------
  // Compute word index from byte offset
  // ------------------------------------------------------------
  int byteOffsetToWordIndex(const std::string &s, size_t bytePos) {
    int count = 0;
    for (size_t i = 0; i < bytePos && i < s.size(); ++i)
      if (s[i] == ' ')
        ++count;
    return count;
  }

  void combine_partial_gleason_annotations(std::vector<ConceptSpan> &concepts) {
    auto is_gleason = [](const ConceptSpan &c) {
      return c.concept_type == "gleason_grade";
    };

    auto is_tumor_percent = [](const ConceptSpan &c) {
      return c.concept_type == "tumor_presence" &&
             c.attributes.percent_involvement.has_value();
    };

    auto compatible = [&](const ConceptSpan &a,
                          const ConceptSpan &b) {
      // only merge positive assertions for now
      if (a.assertion != "present" ||
          b.assertion != "present") {
        return false;
      }

      bool allowed_pair =
          (is_gleason(a) && is_gleason(b)) ||
          (is_gleason(a) && is_tumor_percent(b)) ||
          (is_tumor_percent(a) && is_gleason(b));

      if (!allowed_pair) {
        return false;
      }

      const auto &aa = a.attributes;
      const auto &bb = b.attributes;

      auto attr_compatible = [](const auto &x,
                                const auto &y) {
        // conflicting populated attributes
        if (x.has_value() &&
            y.has_value() &&
            x.value() != y.value()) {
          return false;
        }

        return true;
      };

      return
          attr_compatible(aa.gleason_primary,
                          bb.gleason_primary) &&

          attr_compatible(aa.gleason_secondary,
                          bb.gleason_secondary) &&

          attr_compatible(aa.percent_involvement,
                          bb.percent_involvement);
    };

    auto merge_into = [](ConceptSpan &dst,
                         const ConceptSpan &src) {
      auto merge_attr = [](auto &dstAttr,
                           const auto &srcAttr) {
        if (!dstAttr.has_value() &&
            srcAttr.has_value()) {
          dstAttr = srcAttr;
        }
      };

      merge_attr(dst.attributes.gleason_primary,
                 src.attributes.gleason_primary);

      merge_attr(dst.attributes.gleason_secondary,
                 src.attributes.gleason_secondary);

      merge_attr(dst.attributes.percent_involvement,
                 src.attributes.percent_involvement);

      // preserve gleason concept type if either side is gleason
      if (src.concept_type == "gleason_grade") {
        dst.concept_type = "gleason_grade";
      }

      // combine evidence text
      if (!src.evidence_text.empty()) {
        if (!dst.evidence_text.empty())
          dst.evidence_text += " | ";

        dst.evidence_text += src.evidence_text;
      }

      // combine concept text
      if (!src.concept_text.empty()) {
        if (!dst.concept_text.empty())
          dst.concept_text += " | ";

        dst.concept_text += src.concept_text;
      }

      // expand textual span
      if (src.spanStartI != -1) {
        if (dst.spanStartI == -1)
          dst.spanStartI = src.spanStartI;
        else
          dst.spanStartI =
              std::min(dst.spanStartI,
                       src.spanStartI);
      }

      if (src.spanEndI != -1) {
        if (dst.spanEndI == -1)
          dst.spanEndI = src.spanEndI;
        else
          dst.spanEndI =
              std::max(dst.spanEndI,
                       src.spanEndI);
      }

      // expand timestamps
      if (src.startMS != -1) {
        if (dst.startMS == -1)
          dst.startMS = src.startMS;
        else
          dst.startMS =
              std::min(dst.startMS,
                       src.startMS);
      }

      if (src.endMS != -1) {
        if (dst.endMS == -1)
          dst.endMS = src.endMS;
        else
          dst.endMS =
              std::max(dst.endMS,
                       src.endMS);
      }

      // expand frame range
      if (src.startFrameIdx != -1) {
        if (dst.startFrameIdx == -1)
          dst.startFrameIdx =
              src.startFrameIdx;
        else
          dst.startFrameIdx =
              std::min(dst.startFrameIdx,
                       src.startFrameIdx);
      }

      if (src.endFrameIdx != -1) {
        if (dst.endFrameIdx == -1)
          dst.endFrameIdx =
              src.endFrameIdx;
        else
          dst.endFrameIdx =
              std::max(dst.endFrameIdx,
                       src.endFrameIdx);
      }

      dst.slideLevel =
          dst.slideLevel || src.slideLevel;
    };

    std::vector<bool> removed(concepts.size(),
                              false);

    for (size_t i = 0;
         i < concepts.size();
         ++i) {
      if (removed[i])
        continue;

      for (size_t j = i + 1;
           j < concepts.size();
           ++j) {
        if (removed[j])
          continue;

        if (!compatible(concepts[i],
                        concepts[j])) {
          continue;
        }

        const auto &a =
            concepts[i].attributes;

        const auto &b =
            concepts[j].attributes;

        bool complementary = false;

        auto contributes_missing =
            [](const auto &x,
               const auto &y) {
          return !x.has_value() &&
                 y.has_value();
        };

        complementary |=
            contributes_missing(
              a.gleason_primary,
              b.gleason_primary);

        complementary |=
            contributes_missing(
              a.gleason_secondary,
              b.gleason_secondary);

        complementary |=
            contributes_missing(
              a.percent_involvement,
              b.percent_involvement);

        complementary |=
            contributes_missing(
              b.gleason_primary,
              a.gleason_primary);

        complementary |=
            contributes_missing(
              b.gleason_secondary,
              a.gleason_secondary);

        complementary |=
            contributes_missing(
              b.percent_involvement,
              a.percent_involvement);

        if (!complementary) {
          continue;
        }

        merge_into(concepts[i],
                   concepts[j]);

        removed[j] = true;
      }
    }

    std::vector<ConceptSpan> merged;
    merged.reserve(concepts.size());

    for (size_t i = 0;
         i < concepts.size();
         ++i) {
      if (!removed[i]) {
        merged.push_back(
          std::move(concepts[i]));
      }
    }

    concepts = std::move(merged);
  }

  std::string rebuild_transcript_with_silence_punctuation(
    const std::vector<tsWord>& wordVec,
    long silenceThresholdMS = 2000)
  {
    if (wordVec.empty())
      return "";

    auto ends_with_punctuation =
        [](const std::string& s)
        {
          if (s.empty())
            return false;

          return static_cast<bool>(
              std::ispunct(
                  static_cast<unsigned char>(
                      s.back())));
        };

    std::string result;

    for (size_t i = 0; i < wordVec.size(); ++i) {

      const auto& w = wordVec[i];

      // add space before non-first words
      if (!result.empty())
        result += " ";

      result += w.word;

      // look ahead to next word
      if (i + 1 < wordVec.size()) {

        const auto& next =
            wordVec[i + 1];

        long silence =
            next.startMS - w.endMS;

        if (silence > silenceThresholdMS) {

          if (!ends_with_punctuation(
                  w.word))
          {
            result += ",";
          }
        }
      }
    }

    // ensure transcript ends with punctuation
    if (!result.empty() &&
        !ends_with_punctuation(result))
    {
      result += ".";
    }

    return result;
  }

  void resolveEvidenceSpans(const std::string &originalText,
                          std::vector<ConceptSpan> &annotations) {
  if (originalText.empty())
    return;

  auto stripPunctuation =
      [](const std::string &s) {

        std::string out;
        out.reserve(s.size());

        for (char c: s) {

          if (!std::ispunct(
                  static_cast<unsigned char>(c))) {
            out += c;
          }
        }

        return out;
      };

  auto normalizeLoose =
      [&](const std::string &s) {

        return toLower(
          stripPunctuation(
            normalize(s)));
      };

  size_t searchStartByte = 0;

  // original transcript tokens
  const auto transcriptTokens =
      splitWords(originalText);

  // normalized punctuation-free transcript tokens
  std::vector<std::string> normalizedTranscriptTokens;
  normalizedTranscriptTokens.reserve(
      transcriptTokens.size());

  for (const auto &t: transcriptTokens) {
    normalizedTranscriptTokens.push_back(
        normalizeLoose(t));
  }

  for (auto &ann: annotations) {

    if (ann.evidence_text.empty())
      continue;

    // ---------------------------------------------
    // 1) Exact substring match
    // ---------------------------------------------
    size_t pos =
        originalText.find(
            ann.evidence_text,
            searchStartByte);

    if (pos != std::string::npos) {

      int startWord =
          byteOffsetToWordIndex(
              originalText,
              pos);

      int endWord =
          byteOffsetToWordIndex(
              originalText,
              pos +
              ann.evidence_text.size() - 1);

      ann.spanStartI = startWord;
      ann.spanEndI = endWord;

      searchStartByte =
          pos + ann.evidence_text.size();

      continue;
    }

    // ---------------------------------------------
    // 2) Punctuation-insensitive token match
    // ---------------------------------------------
    const auto evidenceTokens =
        splitWords(ann.evidence_text);

    if (!evidenceTokens.empty()) {

      std::vector<std::string>
          normalizedEvidenceTokens;

      normalizedEvidenceTokens.reserve(
          evidenceTokens.size());

      for (const auto &t: evidenceTokens) {
        normalizedEvidenceTokens.push_back(
            normalizeLoose(t));
      }

      const size_t tSize =
          normalizedTranscriptTokens.size();

      const size_t eSize =
          normalizedEvidenceTokens.size();

      for (size_t i = 0;
           i + eSize <= tSize;
           ++i) {

        bool match = true;

        for (size_t j = 0;
             j < eSize;
             ++j) {

          if (normalizedTranscriptTokens[i + j] !=
              normalizedEvidenceTokens[j]) {
            match = false;
            break;
          }
        }

        if (match) {

          ann.spanStartI =
              static_cast<int>(i);

          ann.spanEndI =
              static_cast<int>(
                  i + eSize - 1);

          // advance search start
          size_t bytePos = 0;
          int wordCount = 0;

          while (bytePos < originalText.size() &&
                 wordCount <
                 ann.spanEndI + 1) {

            if (originalText[bytePos] == ' ')
              ++wordCount;

            ++bytePos;
          }

          searchStartByte = bytePos;
          break;
        }
      }
    }

    // if all methods fail,
    // indices remain -1
  }
}

  std::vector<std::string> load_andrew_transcriptions() {
    std::ifstream file("/home/cm/Documents/data/Andrew_data_march/transcriptions.txt");
    std::vector<std::string> out;
    std::string line;
    while (std::getline(file, line)) {
      // skip index lines (pure digits) and blank lines
      if (line.empty() || line.find_first_not_of("0123456789") == std::string::npos)
        continue;
      out.push_back(line);
    }
    return out;
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
}

std::vector<std::string> AnnotateComponent::get_preconfig_anno() const {
  if (parent && parent->sCam) {
    return parent->sCam->get_preconfig_anno_labels();
  }
  return {
    "Gleason 3+3", "Gleason 3+4", "Gleason 4+3", "Gleason 4+4", "Gleason 4+5", "Gleason 5+4",
    "positive surgical margin", "seminal vesicle invasion", "perineural invasion", "extraprostatic extension"
  };
}


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


void AnnotateComponent::resized() {
  auto area = getLocalBounds();
  juce::Component *components[] = {leftComponent.get(), resizerBar.get(), rightComponent.get()};

  layout.layOutComponents(components, 3, area.getX(), area.getY(), area.getWidth(), area.getHeight(), false, true);

  auto leftBounds = leftComponent->getBounds();
  int navHeight = leftBounds.getHeight() / 4;
  leftComponent->setBounds(leftBounds.withTrimmedBottom(navHeight));
  navPathList->setBounds(leftBounds.removeFromBottom(navHeight));

  rightComponent->resized();

  parent->repositionSlideListButton();
}

void AnnotateComponent::setImage(std::shared_ptr<MRTiledImageSet> image) {
  ephemeralNavPath.reset();
  getListComp()->setDistancePerFrame(std::nullopt);
  getListComp()->setPathSectionDist(std::nullopt);
  rightComponent->setImage(image);
  if (image) {
    update_active_annotations(image->index);
    if (image->navPaths.empty()) {
      image->generate_nav_paths();
    }
    getNavPathList()->setPaths(image->navPaths);
  }
}


static std::string buildConceptExtractionRequestBody_JSON_Llama(const juce::String &text) {
  juce::String systemMsg =
      "You extract atomic pathology concepts from slide-review transcripts.\n"
      "Input: a single string 'text'.\n"
      "\n"
      "GOAL:\n"
      "Decompose the transcript into the SMALLEST CLINICALLY MEANINGFUL pathology concepts.\n"
      "\n"
      "For each concept, output:\n"
      "- evidence_text: the EXACT contiguous substring from the transcript\n"
      "- concept_text: a concise interpretation (1 to 10 words, NOT a final label)\n"
      "- concept_type: one of ['inflammation','invasion','margin','gleason_grade','extraprostatic_extension','carcinoma','architecture','other']\n"
      "- assertion: one of ['present','absent','uncertain','revised']\n"
      "- attributes: object (may be empty)\n"
      "\n"
      "CRITICAL RULES (STRICT):\n"
      "1) evidence_text MUST be copied VERBATIM from the transcript.\n"
      "2) evidence_text MUST be a SINGLE contiguous substring.\n"
      "3) DO NOT paraphrase or modify evidence_text.\n"
      "4) Extract ALL concepts (INCLUDING negative ones).\n"
      "5) DO NOT skip concepts because another seems more important.\n"
      "\n"
      "6) CONCEPT GRANULARITY (VERY IMPORTANT):\n"
      "   Extract the smallest CLINICALLY MEANINGFUL concepts, NOT the smallest phrases.\n"
      "\n"
      "   The following MUST remain grouped as a SINGLE concept:\n"
      "\n"
      "   a) GLEASON GROUPING:\n"
      "      Any Gleason score MUST include ALL directly associated information:\n"
      "      - Gleason pattern (e.g. 3+4)\n"
      "      - Percent involvement (if present)\n"
      "      - Immediate continuation phrases\n"
      "\n"
      "      Example:\n"
      "      'Gleason 3 plus 3 equals 6, involving 2% of prostate present'\n"
      "      -> ONE concept\n"
      "\n"
      "   b) MARGIN GROUPING:\n"
      "      Margin statements MUST include the governing term 'margin'.\n"
      "\n"
      "      Example:\n"
      "      'Margin negative for tumor'\n"
      "      -> evidence_text MUST include 'Margin'\n"
      "\n"
      "   c) GOVERNING NOUN RULE:\n"
      "      If a phrase depends on a governing term (e.g. margin, invasion, glands),\n"
      "      the governing term MUST be included in evidence_text.\n"
      "\n"
      "   d) DEPENDENT PHRASES:\n"
      "      Do NOT split a concept if a later phrase depends on an earlier one.\n"
      "\n"
      "      BAD:\n"
      "        'Gleason 3+3' + 'involving 2%'\n"
      "      GOOD:\n"
      "        one combined concept\n"
      "\n"
      "   e) Prefer slightly larger spans over fragmented ones when meaning would be lost.\n"
      "\n"
      "7) SPLIT truly independent concepts:\n"
      "   Example: 'acute and chronic inflammation' -> TWO concepts.\n"
      "\n"
      "8) DO NOT merge separate independent findings into one.\n"
      "9) concept_text MUST be derived ONLY from evidence_text.\n"
      "10) DO NOT use context outside the evidence_text.\n"
      "\n"
      "11) 'prostatic adenocarcinoma' is context and SHOULD NOT be a standalone concept\n"
      "    unless it is the ONLY finding in the transcript.\n"
      "\n"
      "12) Assertion mapping:\n"
      "   - 'positive', 'present' -> present\n"
      "   - 'negative for', 'no', 'absent' -> absent\n"
      "   - 'maybe', 'possible', 'cannot exclude' -> uncertain\n"
      "   - corrections -> revised\n"
      "\n"
      "13) SPAN PRECISION:\n"
      "   evidence_text must contain ONLY the words expressing the concept.\n"
      "   Do NOT include neighboring concepts.\n"
      "\n"
      "14) ORDER:\n"
      "   Maintain original order of appearance.\n"
      "\n"
      "15) GLEASON EXTRACTION:\n"
      "   For Gleason, extract attributes:\n"
      "   - gleason_primary\n"
      "   - gleason_secondary\n"
      "   - percent_involvement (if present)\n"
      "\n"
      "16) SELF-CHECK BEFORE OUTPUT:\n"
      "   - Every evidence_text appears EXACTLY in the input string\n"
      "   - No concepts missing\n"
      "   - No incorrectly split grouped concepts\n"
      "\n"
      "17) OUTPUT FORMAT (MANDATORY):\n"
      "   You MUST output ONLY valid JSON.\n"
      "   Do NOT include explanations.\n"
      "   Do NOT include markdown.\n"
      "   Do NOT include any text before or after the JSON.\n"
      "\n"
      "18) JSON STRUCTURE (STRICT):\n"
      "   The output MUST EXACTLY match this schema:\n"
      "\n"
      "   {\n"
      "     \"concepts\": [\n"
      "       {\n"
      "         \"evidence_text\": string,\n"
      "         \"concept_text\": string,\n"
      "         \"concept_type\": string,\n"
      "         \"assertion\": string,\n"
      "         \"attributes\": {\n"
      "           \"gleason_primary\": number or null,\n"
      "           \"gleason_secondary\": number or null,\n"
      "           \"percent_involvement\": number or null\n"
      "         }\n"
      "       }\n"
      "     ]\n"
      "   }\n"
      "\n"
      "19) FAILURE CASE:\n"
      "   If no concepts are found, return EXACTLY:\n"
      "   {\"concepts\":[]}\n"
      "\n"
      "20) FINAL RULE:\n"
      "   Your response MUST be parseable by a strict JSON parser with no modifications.\n";

  juce::DynamicObject::Ptr root(new juce::DynamicObject());

  root->setProperty("model", "local-llama"); // ignored by llama.cpp but keep for compatibility
  root->setProperty("temperature", 0);

  // ---- messages array ----
  juce::Array<juce::var> messages;

  // system message
  {
    juce::DynamicObject::Ptr sys(new juce::DynamicObject());
    sys->setProperty("role", "system");
    sys->setProperty("content", systemMsg);
    messages.add(juce::var(sys.get()));
  }

  // user message
  {
    juce::DynamicObject::Ptr usr(new juce::DynamicObject());
    usr->setProperty("role", "user");
    usr->setProperty("content", text);
    messages.add(juce::var(usr.get()));
  }

  root->setProperty("messages", juce::var(messages));

  // optional but useful: stop generation drift
  juce::Array<juce::var> stopArr;
  stopArr.add("\n\n");
  // root->setProperty("stop", juce::var(stopArr));

  // serialize
  return juce::JSON::toString(juce::var(root.get()), true).toStdString();
}


static std::string buildConceptExtractionRequestBody_JSON(const juce::String &text) {
  juce::String systemMsg =
      "You extract atomic pathology concepts from slide-review transcripts.\n"
      "Input: a single string 'text'.\n"
      "\n"
      "GOAL:\n"
      "Decompose the transcript into the SMALLEST CLINICALLY MEANINGFUL pathology concepts.\n"
      "\n"
      "For each concept, output:\n"
      "- evidence_text: the EXACT contiguous substring from the transcript\n"
      "- concept_text: a concise interpretation (1 to 10 words, NOT a final label)\n"
      "- concept_type: one of ['inflammation','invasion_pattern','margin_status','gleason_grade','extension_pattern','tumor_presence','architectural_pattern','other']\n"
      "- assertion: one of ['present','absent','uncertain','revised']\n"
      "- attributes: object (may be empty)\n"
      "\n"
      "CRITICAL RULES (STRICT):\n"
      "1) evidence_text MUST be copied VERBATIM from the transcript.\n"
      "2) evidence_text MUST be a SINGLE contiguous substring.\n"
      "3) DO NOT paraphrase or modify evidence_text.\n"
      "4) Extract ALL concepts (INCLUDING negative ones).\n"
      "5) DO NOT skip concepts because another seems more important.\n"
      "\n"
      "6) CONCEPT GRANULARITY (VERY IMPORTANT):\n"
      "   Extract the smallest CLINICALLY MEANINGFUL concepts, NOT the smallest phrases.\n"
      "\n"
      "   The following MUST remain grouped as a SINGLE concept:\n"
      "\n"
      "   a) GLEASON GROUPING:\n"
      "      Any Gleason score MUST include ALL directly associated information:\n"
      "      - Gleason pattern (e.g. 3+4)\n"
      "      - Percent involvement (if present)\n"
      "      - Immediate continuation phrases\n"
      "\n"
      "      Example:\n"
      "      'Gleason 3 plus 3 equals 6, involving 2% of prostate present'\n"
      "      -> ONE concept\n"
      "\n"
      "   b) MARGIN GROUPING:\n"
      "      Margin statements MUST include the governing term 'margin'.\n"
      "\n"
      "      Example:\n"
      "      'Margin negative for tumor'\n"
      "      -> evidence_text MUST include 'Margin'\n"
      "\n"
      "   c) GOVERNING NOUN RULE:\n"
      "      If a phrase depends on a governing term (e.g. margin, invasion, glands),\n"
      "      the governing term MUST be included in evidence_text.\n"
      "\n"
      "   d) DEPENDENT PHRASES:\n"
      "      Do NOT split a concept if a later phrase depends on an earlier one.\n"
      "\n"
      "      BAD:\n"
      "        'Gleason 3+3' + 'involving 2%'\n"
      "      GOOD:\n"
      "        one combined concept\n"
      "\n"
      "   e) Prefer slightly larger spans over fragmented ones when meaning would be lost.\n"
      "\n"
      "7) SPLIT truly independent concepts:\n"
      "   Example: 'acute and chronic inflammation' -> TWO concepts.\n"
      "\n"
      "8) DO NOT merge separate independent findings into one.\n"
      "9) concept_text MUST be derived ONLY from evidence_text.\n"
      "10) DO NOT use context outside the evidence_text.\n"
      "\n"
      "11) 'prostatic adenocarcinoma' is context and SHOULD NOT be a standalone concept\n"
      "    unless it is the ONLY finding in the transcript.\n"
      "\n"
      "12) Assertion mapping:\n"
      "   - 'positive', 'present' -> present\n"
      "   - 'negative for', 'no', 'absent' -> absent\n"
      "   - 'maybe', 'possible', 'cannot exclude' -> uncertain\n"
      "   - corrections -> revised\n"
      "\n"
      "13) SPAN PRECISION:\n"
      "   evidence_text must contain ONLY the words expressing the concept.\n"
      "   Do NOT include neighboring concepts.\n"
      "\n"
      "14) ORDER:\n"
      "   Maintain original order of appearance.\n"
      "\n"
      "15) GLEASON EXTRACTION:\n"
      "   For Gleason, extract attributes:\n"
      "   - gleason_primary\n"
      "   - gleason_secondary\n"
      "   - percent_involvement (if present)\n"
      "\n"
      "16) SELF-CHECK BEFORE OUTPUT:\n"
      "   - Every evidence_text appears EXACTLY in the input string\n"
      "   - No concepts missing\n"
      "   - No incorrectly split grouped concepts\n";

  // helper
  auto makeTypeObj = [](const juce::String &t) -> juce::var {
    juce::DynamicObject::Ptr o(new juce::DynamicObject());
    o->setProperty("type", t);
    return {o.get()};
  };

  // --- attributes object
  juce::DynamicObject::Ptr attrProps(new juce::DynamicObject());

  auto intOrNull = [](const juce::String &t) -> juce::var {
    juce::DynamicObject::Ptr o(new juce::DynamicObject());
    juce::Array<juce::var> types;
    types.add("integer");
    types.add("null");
    o->setProperty("type", juce::var(types));
    return {o.get()};
  };

  auto numOrNull = [](const juce::String &t) -> juce::var {
    juce::DynamicObject::Ptr o(new juce::DynamicObject());
    juce::Array<juce::var> types;
    types.add("number");
    types.add("null");
    o->setProperty("type", juce::var(types));
    return {o.get()};
  };

  attrProps->setProperty("gleason_primary", intOrNull("integer"));
  attrProps->setProperty("gleason_secondary", intOrNull("integer"));
  attrProps->setProperty("percent_involvement", numOrNull("number"));

  juce::Array<juce::var> attrRequired;
  attrRequired.add("gleason_primary");
  attrRequired.add("gleason_secondary");
  attrRequired.add("percent_involvement");

  juce::DynamicObject::Ptr attrObj(new juce::DynamicObject());
  attrObj->setProperty("type", "object");
  attrObj->setProperty("properties", juce::var(attrProps.get()));
  attrObj->setProperty("required", juce::var(attrRequired));
  attrObj->setProperty("additionalProperties", false);

  // --- properties
  juce::DynamicObject::Ptr props(new juce::DynamicObject());
  props->setProperty("evidence_text", makeTypeObj("string"));
  props->setProperty("concept_text", makeTypeObj("string"));
  props->setProperty("concept_type", makeTypeObj("string"));
  props->setProperty("assertion", makeTypeObj("string"));
  props->setProperty("attributes", juce::var(attrObj.get()));

  // required
  juce::Array<juce::var> requiredArr;
  requiredArr.add("evidence_text");
  requiredArr.add("concept_text");
  requiredArr.add("concept_type");
  requiredArr.add("assertion");
  requiredArr.add("attributes");

  juce::DynamicObject::Ptr item(new juce::DynamicObject());
  item->setProperty("type", "object");
  item->setProperty("properties", juce::var(props.get()));
  item->setProperty("required", juce::var(requiredArr));
  item->setProperty("additionalProperties", false);

  juce::DynamicObject::Ptr arrayProp(new juce::DynamicObject());
  arrayProp->setProperty("type", "array");
  arrayProp->setProperty("items", juce::var(item.get()));

  // root schema
  juce::DynamicObject::Ptr rootProps(new juce::DynamicObject());
  rootProps->setProperty("concepts", juce::var(arrayProp.get()));

  juce::Array<juce::var> rootRequired;
  rootRequired.add("concepts");

  juce::DynamicObject::Ptr schema(new juce::DynamicObject());
  schema->setProperty("type", "object");
  schema->setProperty("properties", juce::var(rootProps.get()));
  schema->setProperty("required", juce::var(rootRequired));
  schema->setProperty("additionalProperties", false);

  // format
  juce::DynamicObject::Ptr format(new juce::DynamicObject());
  format->setProperty("type", "json_schema");
  format->setProperty("name", "concept_extraction");
  format->setProperty("strict", true);
  format->setProperty("schema", juce::var(schema.get()));

  juce::DynamicObject::Ptr textObj(new juce::DynamicObject());
  textObj->setProperty("format", juce::var(format.get()));

  // root request
  juce::DynamicObject::Ptr root(new juce::DynamicObject());
  root->setProperty("model", "gpt-4o-mini");
  root->setProperty("instructions", systemMsg);
  root->setProperty("input", text);
  root->setProperty("temperature", 0);
  root->setProperty("text", juce::var(textObj.get()));

  return juce::JSON::toString(juce::var(root.get()), true).toStdString();
}


static std::string LlamaResponses_POST(const std::string &requestBodyJson, int timeoutMs = 120000) {
  juce::URL url("http://127.0.0.1:8081/v1/chat/completions");

  juce::String headers;
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

static std::string openAIResponses_POST(const std::string &apiKey,
                                        const std::string &requestBodyJson,
                                        int timeoutMs = 120000) {
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


static std::vector<ConceptSpan> parseConceptsFromResponses(const std::string &responsesJson) {
  std::vector<ConceptSpan> out;

  auto top = juce::JSON::parse(responsesJson);
  if (!top.isObject()) return out;

  auto *topObj = top.getDynamicObject();
  if (!topObj) return out;

  auto outputVar = topObj->getProperty("output");
  auto *outArr = outputVar.getArray();
  if (!outArr) return out;

  juce::var messageContentVar;

  // ==============================
  // 1) Find message
  // ==============================
  for (const auto &item: *outArr) {
    auto *msgObj = item.getDynamicObject();
    if (!msgObj) continue;

    if (msgObj->getProperty("type").toString() == "message") {
      messageContentVar = msgObj->getProperty("content");
      break;
    }
  }

  auto *msgContentArr = messageContentVar.getArray();
  if (!msgContentArr) return out;

  juce::String payloadText;

  // ==============================
  // Helper: parse concepts array
  // ==============================
  auto parseConceptArray = [&](juce::Array<juce::var> *conceptArr) {
    if (!conceptArr) return;

    out.reserve((size_t) conceptArr->size());

    for (const auto &cv: *conceptArr) {
      auto *cobj = cv.getDynamicObject();
      if (!cobj) continue;

      ConceptSpan c;

      // ---- core fields ----
      c.evidence_text = cobj->getProperty("evidence_text").toString().toStdString();
      c.concept_text = cobj->getProperty("concept_text").toString().toStdString();
      c.concept_type = cobj->getProperty("concept_type").toString().toStdString();
      c.assertion = cobj->getProperty("assertion").toString().toStdString();

      // ---- attributes ----
      auto attrVar = cobj->getProperty("attributes");
      auto *attrObj = attrVar.getDynamicObject();

      if (attrObj) {
        auto gp = attrObj->getProperty("gleason_primary");
        auto gs = attrObj->getProperty("gleason_secondary");
        auto pi = attrObj->getProperty("percent_involvement");

        if (!gp.isVoid())
          c.attributes.gleason_primary = (int) gp;

        if (!gs.isVoid())
          c.attributes.gleason_secondary = (int) gs;

        if (!pi.isVoid())
          c.attributes.percent_involvement = (double) pi;
      }

      if (!c.evidence_text.empty())
        out.push_back(std::move(c));
    }
  };

  // ==============================
  // 2) Extract output_text / json
  // ==============================
  for (const auto &c: *msgContentArr) {
    auto *cobj = c.getDynamicObject();
    if (!cobj) continue;

    const auto ctype = cobj->getProperty("type").toString();

    if (ctype == "output_text") {
      payloadText = cobj->getProperty("text").toString();
      break;
    } else if (ctype == "output_json") {
      auto jsonVar = cobj->getProperty("content");
      auto *jsonObj = jsonVar.getDynamicObject();
      if (!jsonObj) return out;

      auto conceptsVar = jsonObj->getProperty("concepts");
      auto *conceptArr = conceptsVar.getArray();

      parseConceptArray(conceptArr);
      return out;
    }
  }

  if (payloadText.isEmpty())
    return out;

  // ==============================
  // 3) Parse inner JSON string
  // ==============================
  auto inner = juce::JSON::parse(payloadText);
  if (!inner.isObject()) return out;

  auto *innerObj = inner.getDynamicObject();
  if (!innerObj) return out;

  auto conceptsVar = innerObj->getProperty("concepts");
  auto *conceptArr = conceptsVar.getArray();

  parseConceptArray(conceptArr);

  return out;
}

static std::vector<ConceptSpan> parseConceptsFromLlamaResponses(const std::string &responsesJson) {
  std::vector<ConceptSpan> out;

  // ==============================
  // 1) Parse outer response
  // ==============================
  auto top = juce::JSON::parse(responsesJson);
  if (!top.isObject()) return out;

  auto *topObj = top.getDynamicObject();
  if (!topObj) return out;

  auto choicesVar = topObj->getProperty("choices");
  auto *choicesArr = choicesVar.getArray();
  if (!choicesArr || choicesArr->isEmpty()) return out;

  auto *choiceObj = choicesArr->getReference(0).getDynamicObject();
  if (!choiceObj) return out;

  auto messageVar = choiceObj->getProperty("message");
  auto *msgObj = messageVar.getDynamicObject();
  if (!msgObj) return out;

  std::string content = msgObj->getProperty("content").toString().toStdString();
  if (content.empty()) return out;

  // ==============================
  // 2) Extract JSON array from string
  // ==============================
  auto start = content.find("[");
  auto end = content.rfind("]");

  if (start == std::string::npos || end == std::string::npos || end <= start)
    return out;

  std::string jsonArrayStr = content.substr(start, end - start + 1);

  auto parsed = juce::JSON::parse(jsonArrayStr);
  if (!parsed.isArray()) return out;

  auto *conceptArr = parsed.getArray();
  if (!conceptArr) return out;

  // ==============================
  // 3) Parse concepts directly
  // ==============================
  out.reserve((size_t) conceptArr->size());

  for (const auto &cv: *conceptArr) {
    auto *cobj = cv.getDynamicObject();
    if (!cobj) continue;

    ConceptSpan c;

    c.evidence_text = cobj->getProperty("evidence_text").toString().toStdString();
    c.concept_text = cobj->getProperty("concept_text").toString().toStdString();
    c.concept_type = cobj->getProperty("concept_type").toString().toStdString();
    c.assertion = cobj->getProperty("assertion").toString().toStdString();

    // ---- attributes ----
    auto attrVar = cobj->getProperty("attributes");
    auto *attrObj = attrVar.getDynamicObject();

    if (attrObj) {
      auto gp = attrObj->getProperty("gleason_primary");
      auto gs = attrObj->getProperty("gleason_secondary");
      auto pi = attrObj->getProperty("percent_involvement");

      if (!gp.isVoid()) c.attributes.gleason_primary = (int) gp;
      if (!gs.isVoid()) c.attributes.gleason_secondary = (int) gs;
      if (!pi.isVoid()) c.attributes.percent_involvement = (double) pi;
    }

    // ==============================
    // 4) Fix assertion deterministically
    // ==============================
    std::string lower = c.evidence_text;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower.find("negative") != std::string::npos ||
        lower.find("no ") != std::string::npos) {
      c.assertion = "absent";
    } else if (lower.find("positive") != std::string::npos ||
               lower.find("present") != std::string::npos) {
      c.assertion = "present";
    }

    // Gleason ALWAYS wins
    if (lower.find("gleason") != std::string::npos &&
        (c.attributes.gleason_primary.has_value() ||
         c.attributes.gleason_secondary.has_value())) {
      c.concept_type = "gleason_grade";
    }

    if (!c.evidence_text.empty())
      out.push_back(std::move(c));
  }

  return out;
}


std::vector<ConceptSpan> reduceToConcepts_LLM(const std::string &text) {
  std::string body = buildConceptExtractionRequestBody_JSON_Llama(text);
  // std::string body = buildConceptSpanExtractionRequestBody_JSON_Llama(text);
  const std::string resp = LlamaResponses_POST(body);

  if (resp.empty()) {
    return {};
  }

  auto pResp = parseConceptsFromLlamaResponses(resp);
  return pResp;
}

std::vector<ConceptSpan> reduceToAnnotations_LLM(const std::string &text,
                                                 const std::vector<std::string> &preconfigAnnos) {

  bool useLocalLLM = false;

  if (useLocalLLM) {
    auto resp = reduceToConcepts_LLM(text);
    resolveEvidenceSpans(text, resp);
    return resp;
  } else {
    std::string body = buildConceptExtractionRequestBody_JSON(text);
    const std::string resp = openAIResponses_POST(apiKey, body);

    if (resp.empty()) {
      return {};
    }

    auto pResp = parseConceptsFromResponses(resp);
    resolveEvidenceSpans(text, pResp);
    return pResp;
    // resolveEvidenceSpans(text, pResp);
    // return pResp;
  }
}


void AnnotateComponent::saveConceptSpans(const std::string &filepath,
                                         const std::vector<ConceptSpan> &spans) {
  juce::Array<juce::var> arr;
  arr.ensureStorageAllocated((int) spans.size());

  for (const auto &s: spans) {
    juce::DynamicObject::Ptr obj(new juce::DynamicObject());
    obj->setProperty("evidence_text", juce::String(s.evidence_text));
    obj->setProperty("concept_text", juce::String(s.concept_text));
    obj->setProperty("concept_type", juce::String(s.concept_type));
    obj->setProperty("assertion", juce::String(s.assertion));
    obj->setProperty("slide_level", s.slideLevel);
    obj->setProperty("span_start_i", s.spanStartI);
    obj->setProperty("span_end_i", s.spanEndI);
    obj->setProperty("start_ms", (juce::int64) s.startMS);
    obj->setProperty("end_ms", (juce::int64) s.endMS);
    obj->setProperty("start_frame_idx", (juce::int64) s.startFrameIdx);
    obj->setProperty("end_frame_idx", (juce::int64) s.endFrameIdx);

    juce::DynamicObject::Ptr attr(new juce::DynamicObject());
    attr->setProperty("gleason_primary", s.attributes.gleason_primary.has_value()
                                           ? juce::var(*s.attributes.gleason_primary)
                                           : juce::var());
    attr->setProperty("gleason_secondary", s.attributes.gleason_secondary.has_value()
                                             ? juce::var(*s.attributes.gleason_secondary)
                                             : juce::var());
    attr->setProperty("percent_involvement", s.attributes.percent_involvement.has_value()
                                               ? juce::var(*s.attributes.percent_involvement)
                                               : juce::var());
    obj->setProperty("attributes", juce::var(attr.get()));

    arr.add(juce::var(obj.get()));
  }

  juce::DynamicObject::Ptr root(new juce::DynamicObject());
  root->setProperty("concept_spans", juce::var(arr));

  juce::File f(filepath);
  f.replaceWithText(juce::JSON::toString(juce::var(root.get()), false));
}

std::vector<ConceptSpan> AnnotateComponent::loadConceptSpans(const std::string &filepath) {
  std::vector<ConceptSpan> out;

  juce::File f(filepath);
  if (!f.existsAsFile())
    return out;

  auto top = juce::JSON::parse(f.loadFileAsString());
  if (!top.isObject())
    return out;

  auto *topObj = top.getDynamicObject();
  if (!topObj)
    return out;

  auto *arr = topObj->getProperty("concept_spans").getArray();
  if (!arr)
    return out;

  out.reserve((size_t) arr->size());

  for (const auto &v: *arr) {
    auto *o = v.getDynamicObject();
    if (!o) continue;

    ConceptSpan s;
    s.evidence_text = o->getProperty("evidence_text").toString().toStdString();
    s.concept_text = o->getProperty("concept_text").toString().toStdString();
    s.concept_type = o->getProperty("concept_type").toString().toStdString();
    s.assertion = o->getProperty("assertion").toString().toStdString();
    s.slideLevel = (bool) o->getProperty("slide_level");
    s.spanStartI = (int) o->getProperty("span_start_i");
    s.spanEndI = (int) o->getProperty("span_end_i");
    s.startMS = (long) (juce::int64) o->getProperty("start_ms");
    s.endMS = (long) (juce::int64) o->getProperty("end_ms");
    s.startFrameIdx = (long) (juce::int64) o->getProperty("start_frame_idx");
    s.endFrameIdx = (long) (juce::int64) o->getProperty("end_frame_idx");

    auto *attr = o->getProperty("attributes").getDynamicObject();
    if (attr) {
      auto gp = attr->getProperty("gleason_primary");
      auto gs = attr->getProperty("gleason_secondary");
      auto pi = attr->getProperty("percent_involvement");
      if (!gp.isVoid()) s.attributes.gleason_primary = (int) gp;
      if (!gs.isVoid()) s.attributes.gleason_secondary = (int) gs;
      if (!pi.isVoid()) s.attributes.percent_involvement = (double) pi;
    }

    out.push_back(std::move(s));
  }

  return out;
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

    //4 huge function cals in a row

    //1) send transcribe to whisper
    auto [fullText,wordVec] = send_transcribe_call(dictPath);

    //2) add punctuation before 2 second silences.
    fullText = rebuild_transcript_with_silence_punctuation(wordVec);

    //3) send transcript to an llm to have the atomic concepts extracted
    auto annoSpanVec = reduceToAnnotations_LLM(fullText, get_preconfig_anno());

    //4) add gleason grades to tumor involvement annotations
    combine_partial_gleason_annotations(annoSpanVec);

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).
    count();

    std::cout << "full text: \"" << fullText << "\" processed in " << dur << std::endl;
    for (auto &annospan: annoSpanVec) {
      std::cout << annospan.concept_type;

      if (annospan.spanStartI >= 0 && annospan.spanEndI >= annospan.spanStartI) {
        std::cout << ", " << annospan.spanStartI << ", " << annospan.spanEndI << ", \"";
        for (int i = annospan.spanStartI; i <= annospan.spanEndI; ++i) {
          std::cout << wordVec[i].word;
          if (i < annospan.spanEndI) {
            std::cout << " ";
          }
        }
        std::cout << "\"" << std::endl;
      }

      if (!annospan.evidence_text.empty()) {
        std::cout << ", \"" << annospan.evidence_text << "\"" << std::endl;
      }
    }
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
    for (auto &conceptSpan: annoSpanVec) {
      if (conceptSpan.spanStartI < 0 || conceptSpan.spanEndI < 0) {
        std::cout << "bad text indices for annotation: " + conceptSpan.concept_type << std::endl;
        continue;
      }

      conceptSpan.startMS = wordVec[conceptSpan.spanStartI].startMS;
      conceptSpan.endMS = wordVec[conceptSpan.spanEndI].endMS;

      long startFrameIndex, endFrameIndex;
      auto polyAnnoVertices = mrImgSet->poly_annotation_from_time_interval(conceptSpan.startMS,
                                                                           conceptSpan.endMS,
                                                                           startFrameIndex,
                                                                           endFrameIndex);

      conceptSpan.startFrameIdx = startFrameIndex;
      conceptSpan.endFrameIdx = endFrameIndex;

      // Create a new polygon annotation with the label
      auto polyAnno = std::make_shared<VoicePointPoly>(conceptSpan);
      std::cout << "annospan frame index: " << conceptSpan.startFrameIdx << " " << conceptSpan.endFrameIdx << std::endl;

      // Add each vertex from polyAnnoVertices
      for (const auto &vertex: polyAnnoVertices) {
        polyAnno->direct_add(fPoint(vertex.x, vertex.y));
      }

      if (conceptSpan.concept_type == "gleason_grade") {
        polyAnno->global = true;
      }

      // Add the polygon annotation to this slide's annotations
      thisSlidesAnnotations->push_back(polyAnno);
    }

    auto dictDir = dictPath.getParentDirectory();
    auto conceptSpanPath = dictDir.getChildFile("conceptSpan");
    saveConceptSpans(conceptSpanPath.getFullPathName().toStdString(), annoSpanVec);

    // Update the list box if this is the current slide
    if (thisSlidesAnnotations == activeAnnotations) {
      leftComponent->requestListRefresh();
      MessageManager::callAsync([this] { rightComponent->repaint(); });
    }
  }

  newVoiceAnnotation.wait();
}

void AnnotateComponent::build_poly_span_annotations_from_save(const std::string &filepath,
                                                              const std::shared_ptr<MRTiledImageSet> &mrImgSet) {
  juce::File csPath(filepath);
  assert(csPath.existsAsFile());

  auto annoSpanVec = loadConceptSpans(filepath);

  //this lock doesnt really need to be done because this vector is only accessed from one thread during load
  Poco::FastMutex::ScopedLock lock(allSlideAnnotationMutex);

  allSlideAnnotations[mrImgSet->index]->reserve(annoSpanVec.size());
  for (auto &cs: annoSpanVec) {
    auto polyAnnoVertices = mrImgSet->poly_annotations_from_frame_interval(cs.startFrameIdx, cs.endFrameIdx);
    auto polyAnno = std::make_shared<VoicePointPoly>(cs);

    // Add each vertex from polyAnnoVertices
    for (const auto &vertex: polyAnnoVertices) {
      polyAnno->direct_add(fPoint(vertex.x, vertex.y));
    }

    if (cs.concept_type == "gleason_grade" || cs.assertion == "absent") {
      polyAnno->global = true;
    }

    allSlideAnnotations[mrImgSet->index]->push_back(polyAnno);
  }
  int k = 0;
}


static std::string buildCAPQuestionRequestBody(
    const std::string& question,
    const std::string& transcriptAnnotations)
{
  juce::DynamicObject::Ptr root =
      new juce::DynamicObject();

  root->setProperty("model", "gpt-4o-mini");

  juce::Array<juce::var> inputArray;

  juce::DynamicObject::Ptr msg =
      new juce::DynamicObject();

  msg->setProperty("role", "user");

  std::string prompt =
      "You are assisting with drafting a preliminary CAP "
      "radical prostatectomy pathology report.\n\n"

      "You will be given:\n"
      "1. A CAP report question\n"
      "2. A set of numbered raw pathology dictation transcripts\n\n"

      "Answer the question to the best of your ability using "
      "ONLY the information in the transcripts.\n\n"

      "If the answer is inferred rather than explicitly stated, "
      "say that clearly.\n\n"

      "At the end of your answer, provide the transcript numbers "
      "used as evidence.\n\n"

      "Use this format:\n\n"

      "Answer: <answer>\n"
      "Evidence: <comma separated transcript numbers>\n\n"

      "Question:\n" +
      question +
      "\n\n"

      "Transcripts:\n" +
      transcriptAnnotations;

  msg->setProperty("content", juce::String(prompt));

  inputArray.add(juce::var(msg));

  root->setProperty("input", inputArray);

  return juce::JSON::toString(juce::var(root))
      .toStdString();
}

static std::string parseResponsesAPIText(
    const std::string& responseJson)
{
  juce::var parsed =
      juce::JSON::parse(responseJson);

  if (parsed.isVoid() ||
      !parsed.isObject()) {
    return {};
      }

  auto* root =
      parsed.getDynamicObject();

  if (root == nullptr)
    return {};

  auto outputVar =
      root->getProperty("output");

  if (!outputVar.isArray())
    return {};

  auto* outputArray =
      outputVar.getArray();

  if (outputArray == nullptr)
    return {};

  for (const auto& outputItem : *outputArray) {

    if (!outputItem.isObject())
      continue;

    auto* outputObj =
        outputItem.getDynamicObject();

    if (outputObj == nullptr)
      continue;

    auto type =
        outputObj->getProperty("type")
            .toString();

    if (type != "message")
      continue;

    auto contentVar =
        outputObj->getProperty("content");

    if (!contentVar.isArray())
      continue;

    auto* contentArray =
        contentVar.getArray();

    if (contentArray == nullptr)
      continue;

    for (const auto& contentItem :
         *contentArray) {

      if (!contentItem.isObject())
        continue;

      auto* contentObj =
          contentItem
              .getDynamicObject();

      if (contentObj == nullptr)
        continue;

      auto contentType =
          contentObj
              ->getProperty("type")
              .toString();

      if (contentType !=
          "output_text")
        continue;

      return contentObj
          ->getProperty("text")
          .toString()
          .toStdString();
         }
  }

  return {};
}
void AnnotateComponent::silly_test() {

  std::string question =
R"(TUMOR

Histologic Type (select all that apply)

Possible answers:

- Acinar adenocarcinoma, conventional (usual)
- Acinar adenocarcinoma, signet-ring-like cell
- Acinar adenocarcinoma, pleomorphic giant cell
- Acinar adenocarcinoma, sarcomatoid
- Acinar adenocarcinoma, prostatic intraepithelial neoplasia-like
- Intraductal carcinoma
- Ductal adenocarcinoma
- Adenosquamous carcinoma
- Squamous cell carcinoma
- Basal cell (adenoid cystic) carcinoma
- Adenocarcinoma with neuroendocrine differentiation
- Well-differentiated neuroendocrine tumor
- Small cell neuroendocrine carcinoma
- Large cell neuroendocrine carcinoma
- Other histologic type not listed
- Carcinoma, type cannot be determined

Determine which histologic type(s) are supported by the pathology dictations.)";

  std::string annotations =
R"(0: Margin negative for tumor, negative for extraprostatic extension. Chronic inflammation present. Prostatic adenocarcinoma, Gleason 3 plus 3 equals 6, involving 2% of prostate present.

1: Margin negative for cancer, negative for extraprostatic extension. Prostatic adenocarcinoma Gleason 3 plus 3 equals 6 involving 1% of prostate present.

2: Margin, focally positive for cancer. Acute and chronic inflammation. Prostatic adenocarcinoma, Gleason 3 plus 4 equals 7, involving approximately 5% of prostate gland. Negative for cribriform glands. Negative for intraductal carcinoma. Negative for perineural invasion.

3: Positive for perineural invasion, margin positive for carcinoma. Prostatic adenocarcinoma Gleason 3 plus 4 equals 7 involving approximately 40% of prostate gland, negative for extraprostatic extension, cribriform glands present.

4: Seminal vesicle negative for carcinoma. Margin negative for carcinoma. No tumor present on current slide.

5: Margin, negative for carcinoma. Negative for extraprostatic extension. Prostatic adenocarcinoma Gleason 3 plus 4 equals 7. Tumor involves 10% of prostate gland. Negative for cribriform glands. Positive for perineural invasion.

6: Margin negative for carcinoma, negative for extraprostatic extension, chronic inflammation present. No cancer on current slide.

7: Margin, negative for carcinoma. Negative for extraprostatic extension. Prostatic adenocarcinoma Gleason 3 plus 3 equals 6 involving 2% of prostate gland present acute and chronic inflammation.

8: Margin is negative for carcinoma, negative for extraprostatic extension, no tumor present on current slide.

9: Margin negative for carcinoma. Prostatic adenocarcinoma Gleason 3 plus 3 equals 6, involving approximately 50% of prostate tissue present. Negative for perineural invasion. Negative for extraprostatic extension.

10: Margin, negative for carcinoma. Negative for extraprostatic extension. Prostatic adenocarcinoma Gleason 3 plus 3 equals 6, involving less than 1% of prostate gland present.

11: Prostatic adenocarcinoma Gleason 3 plus 3 equals 6 involving 2% of prostate gland present, negative for perineural invasion, negative for extraprostatic extension, margins negative for carcinoma.

12: Prostatic adenocarcinoma, Gleason 3 plus 4 equals 7, involving 10% of prostate gland chronic inflammation.

13: Prostatic adenocarcinoma, Gleason 3 plus 4 equals 7 involving 5% of prostate gland present. Positive for cribriform glands.

14: High-grade prostatic intraepithelial neoplasia, chronic inflammation. Prostatic adenocarcinoma Gleason 3 plus 3 equals 6 involving 1% of prostate gland.)";

  auto start = std::chrono::high_resolution_clock::now();

  auto body = buildCAPQuestionRequestBody(question,annotations);
  const std::string resp = openAIResponses_POST(apiKey, body);
  auto pResp = parseResponsesAPIText(resp);

  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).
count();
  int k = 0;


}
