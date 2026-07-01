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
class StreamCamLabelList;

// One row in the slide-selector list. Shows the label as plain text normally;
// switches to an inline TextEditor when the row is clicked a second time while
// already selected. Committing (Return key or click-away) writes back to
// MRTiledImageSet::labelName and flushes to disk.
class SlideRowComponent : public juce::Component,
                          public juce::TextEditor::Listener
{
public:
  explicit SlideRowComponent(StreamCamLabelList* owner) : owner(owner)
  {
    textEditor.setMultiLine(false);
    textEditor.setReturnKeyStartsNewLine(false);
    textEditor.setScrollbarsShown(false);
    textEditor.setCaretVisible(true);
    textEditor.addListener(this);
    addChildComponent(textEditor);
  }

  void update(int rowNum, const juce::String& label, bool selected, bool editing)
  {
    rowNumber    = rowNum;
    currentLabel = label;
    isSelected   = selected;
    isEditing    = editing;

    if (isEditing) {
      textEditor.setText(currentLabel, false);
      textEditor.setVisible(true);
      textEditor.grabKeyboardFocus();
      textEditor.selectAll();
    } else {
      textEditor.setVisible(false);
    }
    repaint();
  }

  void resized() override { textEditor.setBounds(getLocalBounds().reduced(4, 2)); }

  void paint(juce::Graphics& g) override
  {
    g.fillAll(isSelected ? juce::Colours::lightblue.withAlpha(0.4f)
                         : juce::Colours::transparentBlack);
    if (!isEditing) {
      g.setColour(juce::Colours::white);
      g.setFont((float)getHeight() * 0.55f);
      g.drawText(currentLabel, 8, 0, getWidth() - 16, getHeight(),
                 juce::Justification::centredLeft);
    }
  }

  // Defined after StreamCamLabelList below.
  void mouseDown(const juce::MouseEvent& e) override;
  void textEditorFocusLost(juce::TextEditor& ed) override;
  void textEditorReturnKeyPressed(juce::TextEditor& ed) override;

private:
  StreamCamLabelList* owner;
  juce::TextEditor   textEditor;
  juce::String       currentLabel;
  int  rowNumber = -1;
  bool isSelected = false;
  bool isEditing  = false;
};

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

  // Painting is handled entirely by SlideRowComponent.
  void paintListBoxItem(int, juce::Graphics&, int, int, bool) override {}

  // listBoxItemClicked is superseded by SlideRowComponent::mouseDown → rowMouseDown.
  void listBoxItemClicked(int, const juce::MouseEvent&) override {}

  Component* refreshComponentForRow(int rowNumber, bool /*isRowSelected*/,
                                    Component* existingComponentToUpdate) override
  {
    auto* comp = dynamic_cast<SlideRowComponent*>(existingComponentToUpdate);
    if (!comp) {
      delete existingComponentToUpdate;
      comp = new SlideRowComponent(this);
    }

    if (rowNumber < 0 || rowNumber >= (int)filteredIndices.size())
      return comp;

    int actualIndex = filteredIndices[rowNumber];
    juce::String label;
    if (actualIndex >= 0 && actualIndex < sCam->get_num_slides())
      label = sCam->get_slide_label(actualIndex);

    comp->update(rowNumber, label,
                 rowNumber == selectedRow,
                 rowNumber == editingRow);
    return comp;
  }

  // Called by SlideRowComponent::mouseDown.
  void rowMouseDown(int row, const juce::MouseEvent&)
  {
    if (row < 0 || row >= (int)filteredIndices.size()) return;
    int actualIndex = filteredIndices[row];
    if (actualIndex < 0 || actualIndex >= sCam->get_num_slides()) return;

    if (row == selectedRow) {
      // Second click on the already-selected row — enter editing.
      editingRow = row;
    } else {
      // First click — select the row and load the slide.
      editingRow  = -1;
      selectedRow = row;
      if (onClick)
        onClick(actualIndex, sCam->get_slide_label(actualIndex), matchedByAnnotation[row]);
    }

    listBox.updateContent();
    listBox.repaint();
  }

  // Called by SlideRowComponent on Return key or focus loss.
  void commitEdit(int row, const juce::String& newLabel)
  {
    if (row < 0 || row >= (int)filteredIndices.size()) return;
    int actualIndex = filteredIndices[row];

    std::shared_ptr<MRTiledImageSet> slide;
    {
      Poco::FastMutex::ScopedLock lock(sCam->previousSlidesMutex);
      if (actualIndex >= 0 && actualIndex < (int)sCam->previousSlides.size())
        slide = sCam->previousSlides[actualIndex];
    }

    if (slide && newLabel.toStdString() != slide->labelName) {
      slide->labelName = newLabel.toStdString();
      slide->write_slide_header();
    }

    if (editingRow == row) editingRow = -1;
    filterSlides(searchBox.getText());
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
  int selectedRow = -1;
  int editingRow  = -1;
};

// SlideRowComponent methods that call back into StreamCamLabelList — defined
// here so the full StreamCamLabelList definition is in scope.
inline void SlideRowComponent::mouseDown(const juce::MouseEvent& e)
{
  owner->rowMouseDown(rowNumber, e);
}

inline void SlideRowComponent::textEditorFocusLost(juce::TextEditor& ed)
{
  owner->commitEdit(rowNumber, ed.getText());
}

inline void SlideRowComponent::textEditorReturnKeyPressed(juce::TextEditor& ed)
{
  owner->commitEdit(rowNumber, ed.getText());
}

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
  std::unique_ptr<SvgButton> saveButton;
  std::unique_ptr<juce::FileChooser> fc;

  CaptureComponent *parent;


  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CaptureOverlay)
};


#endif /* CaptureOverlay_h */
