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
  using OnClick = std::function<void (int index, const juce::String& label)>;

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
    filterSlides(searchBox.getText());
  }

  void textEditorTextChanged(juce::TextEditor& editor) override
  {
    filterSlides(editor.getText());
  }

private:
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
      onClick(actualSlideIndex, sCam->get_slide_label(actualSlideIndex));
  }

  void updateFilteredIndices(const juce::String& searchText)
  {
    filteredIndices.clear();

    int numSlides = sCam->get_num_slides();
    for (int i = 0; i < numSlides; i++)
    {
      if (searchText.isEmpty())
      {
        // No search text - show all slides
        filteredIndices.push_back(i);
      }
      else
      {
        // Check if this slide has any annotation matching the search text
        if (slideHasMatchingAnnotation(i, searchText))
        {
          filteredIndices.push_back(i);
        }
      }
    }
  }

  bool slideHasMatchingAnnotation(int slideIndex, const juce::String& searchText);

  void filterSlides(const juce::String& searchText)
  {
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
