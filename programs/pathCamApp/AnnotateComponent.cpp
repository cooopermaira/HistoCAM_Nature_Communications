//
//  AnnotateComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/18/24.
//

#include "JuceHeader.h"

namespace {
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
}

void resolveEvidenceSpans(const std::string &originalText,
                          std::vector<AnnotationSpan> &annotations) {
  if (originalText.empty())
    return;

  size_t searchStartByte = 0;

  // Pre-split transcript tokens once (for token matching)
  const auto transcriptTokens = splitWords(originalText);
  const auto normalizedTranscript = normalize(originalText);

  for (auto &ann: annotations) {
    if (ann.evidenceText.empty())
      continue;

    // ---------------------------------------------
    // 1) Exact substring match
    // ---------------------------------------------
    size_t pos = originalText.find(ann.evidenceText, searchStartByte);

    if (pos != std::string::npos) {
      int startWord = byteOffsetToWordIndex(originalText, pos);
      int endWord = byteOffsetToWordIndex(originalText,
                                          pos + ann.evidenceText.size() - 1);

      ann.spanStartI = startWord;
      ann.spanEndI = endWord;

      searchStartByte = pos + ann.evidenceText.size();
      continue;
    }

    // ---------------------------------------------
    // 2) Normalized exact match
    // ---------------------------------------------
    const std::string normalizedEvidence = normalize(ann.evidenceText);

    size_t normPos = normalizedTranscript.find(
      normalizedEvidence,
      normalize(originalText.substr(0, searchStartByte)).size()
    );

    if (normPos != std::string::npos) {
      // Need to map normalized position back to original
      // Simplest deterministic approach:
      // perform token-based match instead (more reliable mapping)
    }

    // ---------------------------------------------
    // 3) Token sequence match (contiguous)
    // ---------------------------------------------
    const auto evidenceTokens = splitWords(ann.evidenceText);

    if (!evidenceTokens.empty()) {
      const size_t tSize = transcriptTokens.size();
      const size_t eSize = evidenceTokens.size();

      for (size_t i = 0; i + eSize <= tSize; ++i) {
        bool match = true;

        for (size_t j = 0; j < eSize; ++j) {
          if (toLower(transcriptTokens[i + j]) !=
              toLower(evidenceTokens[j])) {
            match = false;
            break;
          }
        }

        if (match) {
          ann.spanStartI = static_cast<int>(i);
          ann.spanEndI = static_cast<int>(i + eSize - 1);

          // advance search start
          // convert word index to byte offset
          size_t bytePos = 0;
          int wordCount = 0;

          while (bytePos < originalText.size() &&
                 wordCount < ann.spanEndI + 1) {
            if (originalText[bytePos] == ' ')
              ++wordCount;
            ++bytePos;
          }

          searchStartByte = bytePos;
          break;
        }
      }
    }

    // If all methods fail, annotation remains with -1 indices
  }
}

std::vector<std::string> AnnotateComponent::get_preconfig_anno() {
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
      "- concept_type: one of ['inflammation','invasion','margin','gleason_grade','tumor','architecture','other']\n"
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


static std::string buildResponsesRequestBody_JSON2(const juce::String &text,
                                                   const std::vector<std::string> &preconfigAnnos) {
  juce::String systemMsg =
      "You convert pathology slide-review transcripts into a sequence of short annotation labels.\n"
      "Input: a single string 'text'.\n"
      "\n"
      "Your task:\n"
      "Extract 1..N distinct POSITIVE pathology findings mentioned in the transcript, in the SAME ORDER they appear.\n"
      "\n"
      "For each finding, output:\n"
      "- label: a 1-5 word canonical pathology phrase.\n"
      "- evidence_text: the exact contiguous substring from the transcript that supports this label.\n"
      "- scope: either 'local' or 'global'.\n"
      "\n"
      "Scope definitions: \n"
      "- local = the finding is tied to a specific slide region, focal area, ROI, side, location, core, nodule, or microscopic subregion being discussed. \n"
      "- global = the finding is stated as an overall specimen-level / case-level / final interpretive conclusion, grading summary, or report-level assessment. \n"
      "\n"
      "Rules:\n"
      "1) Output ONLY JSON matching the provided schema.\n"
      "2) Do NOT merge non-contiguous mentions: if topic A then B then A again, output three annotations.\n"
      "3) label must be a concise semantic reduction of the finding.\n"
      "   Example: transcript contains 'Gleason pattern 3 plus 4' -> label 'Gleason 3+4'.\n"
      "4) evidence_text MUST be copied verbatim from the transcript as a single contiguous substring.\n"
      "   Do not paraphrase it. Do not summarize it.\n"
      "5) evidence_text must include all words necessary to justify the label, including qualifiers, numbers, and context words.\n"
      "6) Revision/uncertainty merge rule (MUST follow):\n"
      "   Only merge multiple mentions into ONE annotation when the transcript explicitly presents them as alternative interpretations\n"
      "   or a correction/revision of the SAME finding, using uncertainty/revision markers such as:\n"
      "   \"maybe\", \"probably\", \"or\", \"versus\", \"favors\", \"could be\", \"cannot exclude\", \"actually\", \"no\", \"never mind\", \"on second thought\".\n"
      "   In that case, output exactly ONE annotation labeled with the FINAL favored interpretation (the last decisive claim in that discussion),\n"
      "   and evidence_text MUST be one contiguous verbatim substring spanning from the first mention through the final conclusion/revision.\n"
      "\n"
      "6a) Distinct-assertion split rule (MUST follow):\n"
      "    If the transcript asserts TWO different interpretations/values as separate findings WITHOUT the uncertainty/revision markers above\n"
      "    (often joined by \"and\", \"also\", \"as well as\", or stated in separate clauses/sentences), then output TWO annotations (one per finding).\n"
      "    Gleason-specific example: \"more Gleason 3 plus 4, and Gleason pattern 4 plus 3\" -> output both \"Gleason 3+4\" and \"Gleason 4+3\".\n"
      "\n"
      "6b) Forbidden adjacent-alternatives pattern (guardrail):\n"
      "    Do NOT output two adjacent annotations for the same finding when one is clearly an alternative/correction of the other per rule 6\n"
      "    (e.g. \"Gleason 3+3\" then \"Gleason 3+4\" with \"maybe/probably\"). If you would, merge them into ONE labeled with the final favored interpretation,\n"
      "    and evidence_text spanning both. Do NOT label with an earlier, less confident alternative if a later, more confident alternative is present.\n"
      "\n"
      "7) Prefer canonical pathology wording (e.g., 'perineural invasion', 'positive surgical margin', 'negative surgical margin', 'Gleason 3+4').\n"
      "8) NEGATIVE FINDINGS - SKIP ENTIRELY (MOST IMPORTANT RULE):\n"
      "   If a finding is negative, absent, or not identified - including any phrasing such as\n"
      "   'negative for', 'no evidence of', 'not identified', 'absent', 'none', 'not seen', 'free of', 'clear of' -\n"
      "   DO NOT output an annotation for it. Omit it completely. This applies even if the finding is named.\n"
      "   Example: 'negative for perineural invasion' -> output NOTHING for this finding.\n"
      "   Example: 'margins are clear' -> output NOTHING.\n"
      "   Only output annotations for findings that are PRESENT and POSITIVE.\n"
      "\n"
      "9) Scope MUST be 'local' when the evidence_text is tied to a particular area such as side, apex, base,\n"
      "   nodule, focus, core, or microscopic subregion, OR when it is a Gleason score tied to a specific region or core.\n"
      "\n"
      "10) Scope MUST be 'global' when the evidence_text states an overall specimen-level or case-level conclusion,\n"
      "    such as overall adenocarcinoma involvement, overall margin status, or a generalized summary statement.\n"
      "    IMPORTANT - Gleason global pattern: when the transcript states a Gleason grade together with a percentage\n"
      "    of prostate gland involvement (e.g. 'Gleason 4 plus 3 constituting 60 percent of the prostate gland',\n"
      "    'Gleason 3+4 involving 40% of the gland'), this IS a specimen-level summary - scope MUST be 'global'.\n"
      "    Label it as the Gleason grade (e.g. 'Gleason 4+3') and include the full phrase with percentage in evidence_text.\n"
      "\n"
      "11) If uncertain between 'local' and 'global', prefer 'local' unless the wording clearly indicates a final overall conclusion.\n"
      "\n"
      "12) Before outputting JSON, perform two self-checks:\n"
      "    a) Confirm no annotation corresponds to a negative or absent finding. Remove any that do.\n"
      "    b) Check for adjacent alternative/correction duplicates (rule 6) and merge as required.\n"
      "13) Label-evidence fidelity (MUST follow):\n"
      "    The label MUST be derivable from the evidence_text alone. Do NOT assign a label based on\n"
      "    context elsewhere in the transcript if that concept does not appear in the evidence_text.\n";
  if (!preconfigAnnos.empty()) {
    systemMsg +=
        "14) If any annotation can be adequately described using one of the following exact labels,\n"
        "   you MUST use that exact label verbatim (case-sensitive) instead of inventing a new phrasing:\n";

    for (const auto &anno: preconfigAnnos) {
      systemMsg += "   - " + juce::String(anno) + "\n";
    }
  }

  // Helper to build {"type": "..."} objects for schema leaf nodes
  auto makeTypeObj = [](const juce::String &t) -> juce::var {
    juce::DynamicObject::Ptr o(new juce::DynamicObject());
    o->setProperty("type", t);
    return {o.get()};
  };

  // --- Build item schema: {label, evidence_text}
  juce::DynamicObject::Ptr annProps(new juce::DynamicObject());
  annProps->setProperty("label", makeTypeObj("string"));
  annProps->setProperty("evidence_text", makeTypeObj("string"));

  juce::DynamicObject::Ptr scopeObj(new juce::DynamicObject());
  scopeObj->setProperty("type", "string");
  juce::Array<juce::var> scopeEnum;
  scopeEnum.add("local");
  scopeEnum.add("global");
  scopeObj->setProperty("enum", scopeEnum);

  annProps->setProperty("scope", juce::var(scopeObj.get()));

  juce::Array<juce::var> annRequiredArr;
  annRequiredArr.add("label");
  annRequiredArr.add("evidence_text");
  annRequiredArr.add("scope");
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

static std::string buildResponsesRequestBody_JSON(const juce::String &text,
                                                  const std::vector<std::string> &preconfigAnnos) {
  juce::String systemMsg =
      "You convert pathology slide-review transcripts into a sequence of short annotation labels.\n"
      "Input: a single string 'text'.\n"
      "Define the transcript word list as: split 'text' on single spaces. Indices refer to this list (0-based).\n"
      "Rules:\n"
      "1) Output ONLY JSON matching the provided schema.\n"
      "2) Produce 1..N annotations in the SAME ORDER the ideas appear in the transcript.\n"
      "3) Each annotation corresponds to a contiguous span of the word list: span_start_word..span_end_word (inclusive).\n"
      "4) Do NOT merge non-contiguous mentions: if topic A then B then A again, output three annotations.\n"
      "5) label must be 1-5 words.\n"
      "6) label is a semantic reduction / canonical phrase for the span. It DOES NOT need to be an exact substring.\n"
      "   Example: span contains 'Gleason pattern 3+4' -> label 'Gleason 3+4'.\n"
      "7) Prefer canonical pathology wording (e.g., 'perineural invasion', 'positive margin', 'Gleason 3+4').\n"
      "8) Choose spans that fully cover the evidence words for the idea (include necessary modifiers like \"3+4\").\n"
      "9) span_start_word through span_end_word MUST include all words that justify the label, including qualifiers, numbers, and context words (e.g. \"pattern\", \"plus\"). Do not select a span shorter than the evidence phrase.\n"
      "10) If multiple alternative interpretations of the SAME finding appear in close proximity (e.g. \"maybe\", \"probably\", \"versus\", \"favors\"), and a later statement clearly resolves or favors one, output ONE annotation covering the entire discussion, labeled with the final favored interpretation.\n"
      "11) Before outputting JSON, verify for each label that the words indexed by span_start_word through span_end_word include all thoughts and discussion attributable to that label. If not, expand the span.";

  if (!preconfigAnnos.empty()) {
    systemMsg +=
        "12) If any annotation can be adequately described using one of the following exact labels,\n"
        "    you MUST use that exact label verbatim (case-sensitive) instead of inventing a new phrasing:\n";

    for (const auto &anno: preconfigAnnos) {
      systemMsg += "    - " + juce::String(anno) + "\n";
    }
  }

  // Helper to build {"type": "..."} objects for schema leaf nodes
  auto makeTypeObj = [](const juce::String &t) -> juce::var {
    juce::DynamicObject::Ptr o(new juce::DynamicObject());
    o->setProperty("type", t);
    return {o.get()};
  };

  // --- Build item schema: {label, span_start_word, span_end_word}
  juce::DynamicObject::Ptr annProps(new juce::DynamicObject());
  annProps->setProperty("label", makeTypeObj("string"));
  annProps->setProperty("span_start_word", makeTypeObj("integer"));
  annProps->setProperty("span_end_word", makeTypeObj("integer"));

  // required: ["label","span_start_word","span_end_word"]  (build array safely)
  juce::Array<juce::var> annRequiredArr;
  annRequiredArr.add("label");
  annRequiredArr.add("span_start_word");
  annRequiredArr.add("span_end_word");
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
  root->setProperty("model", "gpt-5-mini");
  root->setProperty("instructions", systemMsg);
  root->setProperty("input", text);
  // root->setProperty("temperature", 0);
  root->setProperty("text", juce::var(textObj.get()));

  return juce::JSON::toString(juce::var(root.get()), true).toStdString();
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

static std::vector<AnnotationSpan> parseAnnotationsFromResponses(const std::string &responsesJson) {
  std::vector<AnnotationSpan> out;

  auto top = juce::JSON::parse(responsesJson);
  if (!top.isObject()) { return out; }

  auto *topObj = top.getDynamicObject();
  if (!topObj) { return out; }

  auto outputVar = topObj->getProperty("output");
  auto *outArr = outputVar.getArray();
  if (!outArr) { return out; }

  // 1) Find first message
  juce::var messageContentVar;

  for (const auto &item: *outArr) {
    auto *msgObj = item.getDynamicObject();
    if (!msgObj) continue;

    if (msgObj->getProperty("type").toString() == "message") {
      messageContentVar = msgObj->getProperty("content");
      break;
    }
  }

  auto *msgContentArr = messageContentVar.getArray();
  if (!msgContentArr) { return out; }

  juce::String payloadText;

  // ------------------------------------------------------------
  // Helper lambda: parse annotations array into AnnotationSpan
  // ------------------------------------------------------------
  auto parseAnnotationArray = [&](juce::Array<juce::var> *annArr) {
    if (!annArr) return;

    out.reserve((size_t) annArr->size());

    for (const auto &av: *annArr) {
      auto *aobj = av.getDynamicObject();
      if (!aobj) continue;

      AnnotationSpan a;
      a.label = aobj->getProperty("label").toString().toStdString();
      a.scope = aobj->getProperty("scope").toString().toStdString();


      // ---- NEW FLEXIBLE HANDLING ----

      auto spanStartVar = aobj->getProperty("span_start_word");
      auto spanEndVar = aobj->getProperty("span_end_word");
      auto evidenceVar = aobj->getProperty("evidence_text");

      const bool hasSpan =
          !spanStartVar.isVoid() && !spanEndVar.isVoid();

      const bool hasEvidence =
          !evidenceVar.isVoid() &&
          evidenceVar.toString().isNotEmpty();

      if (hasSpan) {
        a.spanStartI = (int) spanStartVar;
        a.spanEndI = (int) spanEndVar;
      } else if (hasEvidence) {
        a.evidenceText = evidenceVar.toString().toStdString();
      }

      if (!a.label.empty()) {
        out.push_back(std::move(a));
      }
    }
  };

  // 2) Extract either output_text or output_json
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

      auto annVar = jsonObj->getProperty("annotations");
      auto *annArr = annVar.getArray();
      parseAnnotationArray(annArr);
      return out;
    }
  }

  if (payloadText.isEmpty()) {
    return out;
  }

  // 3) Parse JSON string inside output_text
  auto inner = juce::JSON::parse(payloadText);
  if (!inner.isObject()) return out;

  auto *innerObj = inner.getDynamicObject();
  if (!innerObj) return out;

  auto annVar = innerObj->getProperty("annotations");
  auto *annArr = annVar.getArray();

  parseAnnotationArray(annArr);

  return out;
}


static std::vector<Concept> parseConceptsFromResponses(const std::string &responsesJson) {
  std::vector<Concept> out;

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

      Concept c;

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

std::vector<Concept> reduceToConcepts_OpenAI(const std::string &text) {
  std::string apiKey(
    "REDACTED_OPENAI_API_KEY");
  std::string body = buildConceptExtractionRequestBody_JSON(text);
  const std::string resp = openAIResponses_POST(apiKey, body, 60000);

  if (resp.empty()) {
    return {};
  }

  auto pResp = parseConceptsFromResponses(resp);
  return pResp;
}

std::vector<AnnotationSpan> reduceToAnnotations_OpenAI(const std::string &text,
                                                       const std::vector<std::string> &preconfigAnnos) {
  std::string apiKey(
    "REDACTED_OPENAI_API_KEY");

  bool extractConceptFirst = true;

  if (extractConceptFirst) {
    std::string body = buildConceptExtractionRequestBody_JSON(text);
    const std::string resp = openAIResponses_POST(apiKey, body, 60000);

    if (resp.empty()) {
      return {};
    }

    auto pResp = parseConceptsFromResponses(resp);
    return {};
  } else {
    std::string body = buildResponsesRequestBody_JSON2(text, preconfigAnnos);
    const std::string resp = openAIResponses_POST(apiKey, body, 60000);

    if (resp.empty()) {
      return {};
    }

    auto pResp = parseAnnotationsFromResponses(resp);
    resolveEvidenceSpans(text, pResp);
    return pResp;
  }
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
    auto annoSpanVec = reduceToAnnotations_OpenAI(fullText, get_preconfig_anno());
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).
        count();

    std::cout << "full text: \"" << fullText << "\" processed in " << dur << std::endl;
    for (auto &annospan: annoSpanVec) {
      std::cout << annospan.label;

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

      if (!annospan.evidenceText.empty()) {
        std::cout << ", \"" << annospan.evidenceText << "\"" << std::endl;
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
      std::cout << "annospan frame index: " << annospan.startFrameIdx << " " << annospan.endFrameIdx << std::endl;

      // Add each vertex from polyAnnoVertices
      for (const auto &vertex: polyAnnoVertices) {
        polyAnno->direct_add(fPoint(vertex.x, vertex.y));
      }

      if (annospan.scope == "global") {
        polyAnno->global = true;
      }

      // Add the polygon annotation to this slide's annotations
      thisSlidesAnnotations->push_back(polyAnno);
    }

    // Update the list box if this is the current slide
    if (thisSlidesAnnotations == activeAnnotations) {
      leftComponent->requestListRefresh();
      MessageManager::callAsync([this] { rightComponent->repaint(); });
    }
  }

  newVoiceAnnotation.wait();
}

void AnnotateComponent::silly_test() {
  for (int i = 6; i < 15; ++i) {
    juce::File dictPath("/home/cm/Documents/data/Andrew_data_march/cap" + std::to_string(i) + "/dictation.wav");
    auto [fullText,wordVec] = send_transcribe_call(dictPath);
    std::cout << std::endl << std::endl << i << std::endl;
    size_t width = 120;
    for (size_t i = 0; i < fullText.size(); i += width) {
      std::cout << fullText.substr(i, width) << "\n";
    }
    auto start = std::chrono::high_resolution_clock::now();

    auto annoSpanVec = reduceToConcepts_OpenAI(fullText);

    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).
        count();

    int k = 0;
  }


  juce::File dictPath("/home/cm/Documents/data/Andrew_data_march/cap1/dictation.wav");

  auto start = std::chrono::high_resolution_clock::now();
  auto [fullText,wordVec] = send_transcribe_call(dictPath);

  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).
      count();
  std::cout << "full text: \"" << fullText << "\" processed in " << dur << std::endl;
  auto annoSpanVec = reduceToAnnotations_OpenAI(fullText, get_preconfig_anno());
  for (auto &annospan: annoSpanVec) {
    std::cout << annospan.label;

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

    // if (!annospan.evidenceText.empty()) {
    //   std::cout << ", \"" << annospan.evidenceText << "\"" << std::endl;
    // }
  }
  int k = 0;
}
