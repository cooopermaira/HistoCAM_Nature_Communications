//
//  CaptureOverlay.h
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef CaptureOverlay_h
#define CaptureOverlay_h

#include "JuceHeader.h"

class CaptureComponent;
class AnnotateComponent;

class StreamCamLabelList : public juce::Component,
                           private juce::ListBoxModel,
                           public juce::TextEditor::Listener
{
public:
  using OnClick = std::function<void (int index, const juce::String& label, bool matchedByAnnotation)>;

  StreamCamLabelList(std::shared_ptr<pathCam::StreamCam> sc, AnnotateComponent* annotateComp, OnClick onClickFn)
      : sCam(sc), annotateComp(annotateComp), onClick(std::move(onClickFn))
  {
    listBox.setModel(this);
    listBox.setRowHeight(24);
    listBox.setMultipleSelectionEnabled(false);
    addAndMakeVisible(listBox);

    // Setup search box
    searchBox.setTextToShowWhenEmpty("Search slides by annotation...", juce::Colours::grey);
    searchBox.setMultiLine(false);
    searchBox.setReturnKeyStartsNewLine(false);
    searchBox.setScrollbarsShown(false);
    searchBox.setCaretVisible(true);
    searchBox.setPopupMenuEnabled(true);
    searchBox.addListener(this);
    addAndMakeVisible(searchBox);

    // Initialize with all slides visible
    updateFilteredIndices("");
  }

  void resized() override
  {
    auto b = getLocalBounds();
    auto searchArea = b.removeFromTop(30);
    searchBox.setBounds(searchArea);
    b.removeFromTop(5); // Add small gap
    listBox.setBounds(b);
  }

  // Call this when StreamCam label list changes.
  void refresh()
  {
    auto val = searchBox.getText().toStdString();
    filterSlides(searchBox.getText());
  }

  void textEditorTextChanged(juce::TextEditor& editor) override
  {
    filterSlides(editor.getText());
  }

  // ListBoxModel
  int getNumRows() override
  {
    return (int)filteredIndices.size();
  }

  void paintListBoxItem(int rowNumber, juce::Graphics& g,
                        int width, int height, bool rowIsSelected) override
  {
    if (rowIsSelected)
      g.fillAll(juce::Colours::lightblue.withAlpha(0.25f));

    if (rowNumber < 0 || rowNumber >= (int)filteredIndices.size())
      return;

    int actualSlideIndex = filteredIndices[rowNumber];
    if (actualSlideIndex < 0 || actualSlideIndex >= sCam->get_num_slides())
      return;

    g.setColour(juce::Colours::white);
    g.setFont((float)height * 0.55f);

    auto text = sCam->get_slide_label(actualSlideIndex);
    g.drawText(text, 8, 0, width - 16, height, juce::Justification::centredLeft);
  }

  void listBoxItemClicked(int row, const juce::MouseEvent&) override
  {
    if (row < 0 || row >= (int)filteredIndices.size())
      return;

    int actualSlideIndex = filteredIndices[row];
    if (actualSlideIndex < 0 || actualSlideIndex >= sCam->get_num_slides())
      return;

    if (onClick)
      onClick(actualSlideIndex, sCam->get_slide_label(actualSlideIndex), matchedByAnnotation[row]);
  }

  void updateFilteredIndices(const juce::String& searchText)
  {
    filteredIndices.clear();
    matchedByAnnotation.clear();

    int numSlides = sCam->get_num_slides();
    for (int i = 0; i < numSlides; i++)
    {
      if (searchText.isEmpty())
      {
        // No search text - show all slides
        filteredIndices.push_back(i);
        matchedByAnnotation.push_back(false);
      }
      else
      {
        // Check if slide label matches
        bool labelMatches = false;
        if (i >= 0 && i < sCam->get_num_slides()) {
          juce::String slideLabel = sCam->get_slide_label(i);
          labelMatches = slideLabel.containsIgnoreCase(searchText);
        }

        // Check if any annotation matches
        bool annotationMatches = slideHasAnnotationMatch(i, searchText);

        // Include slide if either label or annotation matches
        if (labelMatches || annotationMatches)
        {
          filteredIndices.push_back(i);
          // Mark as matched by annotation only if label didn't match but annotation did
          matchedByAnnotation.push_back(!labelMatches && annotationMatches);
        }
      }
    }

    // Sort by label alphabetically
    std::vector<size_t> order(filteredIndices.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return sCam->get_slide_label(filteredIndices[a]) <
             sCam->get_slide_label(filteredIndices[b]);
    });

    std::vector<int> sortedIndices(filteredIndices.size());
    std::vector<bool> sortedMatched(matchedByAnnotation.size());
    for (size_t i = 0; i < order.size(); ++i) {
      sortedIndices[i] = filteredIndices[order[i]];
      sortedMatched[i] = matchedByAnnotation[order[i]];
    }
    filteredIndices = std::move(sortedIndices);
    matchedByAnnotation = std::move(sortedMatched);
  }

  bool slideHasMatchingAnnotation(int slideIndex, const juce::String& searchText);
  bool slideHasAnnotationMatch(int slideIndex, const juce::String& searchText);

  juce::String getCurrentSearchText() const { return searchBox.getText(); }

  void filterSlides(const juce::String& searchText)
  {
    // if (searchText.isEmpty()){return;}
    updateFilteredIndices(searchText);
    listBox.updateContent();
    listBox.repaint();
  }

  // Optional: double click behavior instead of single click
  void listBoxItemDoubleClicked(int row, const juce::MouseEvent&) override
  {
    // If you prefer double-click actions, move the callback here.
  }

private:
  std::shared_ptr<pathCam::StreamCam> sCam;
  AnnotateComponent* annotateComp;
  OnClick onClick;

  juce::ListBox listBox { "streamcam labels", this };
  juce::TextEditor searchBox;
  std::vector<int> filteredIndices;
  std::vector<bool> matchedByAnnotation;
};

class CaptureOverlay final : public Component,
                             public Button::Listener {
public:
  CaptureOverlay(CaptureComponent *parent,
                 StringArray &iconNames,
                 OwnedArray<Drawable> &iconsFromZipFile);

  ~CaptureOverlay() {
  }

  void resized() override;

private:
  void buttonClicked(juce::Button *button) override;

  std::unique_ptr<SvgButton> recordButton;
  std::unique_ptr<SvgButton> stopButton;
  std::unique_ptr<SvgButton> simulateButton;

  CaptureComponent *parent;


  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CaptureOverlay)
};


#endif /* CaptureOverlay_h */
