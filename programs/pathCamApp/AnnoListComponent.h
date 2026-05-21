//
//  AnnoListComponent.h
//  pathCamApp
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef AnnoListComponent_h
#define AnnoListComponent_h

#include "JuceHeader.h"

class AnnotateComponent;
class AnnoListBoxModel;

class AnnoListBox : public juce::ListBox
{
public:
  AnnoListBox(AnnotateComponent* parent): parent(parent) {}
  
  void mouseDown(const juce::MouseEvent& event) override;
  
private:
  AnnotateComponent *parent;
};




class AnnoListBoxModel : public ListBoxModel {
  friend class AnnoListComponent;
  friend class ListComponent;
  
public:
  
  AnnoListBoxModel() {  }
  
  int getNumRows() override {
    return (int)(*annotations).size();
  }
  
  void listBoxItemClicked (int row, const MouseEvent&) override;
  
  void listBoxItemDoubleClicked (int row, const MouseEvent&) override;
  
  void backgroundClicked (const MouseEvent&) override;
  
  void paintListBoxItem(int rowNumber, Graphics& g, int width, int height, bool rowIsSelected) override;
  
  Component* refreshComponentForRow (int rowNumber,
                                     bool isRowSelected,
                                     Component* existingComponentToUpdate) override;
  
  void removeSelected();
  
private:
  juce::Drawable * polygonIcon;
  juce::Drawable * measureIcon;
  juce::Drawable * segmentIcon;
  juce::Drawable * trashIcon;
  
  
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  AnnotateComponent *parent;
    
};

class AnnoListComponent : public Component, public juce::TextEditor::Listener, private juce::AsyncUpdater {
public:

  AnnoListComponent(AnnotateComponent *parent,
                    std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > _annotations,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile) : listBox(AnnoListBox(parent)), parent(parent), annotations(_annotations) {

    // Initialize filtered annotations to show all annotations initially
    filteredAnnotations = _annotations;

    model.reset(new AnnoListBoxModel());
    model->annotations = filteredAnnotations;
    model->parent = parent;

    listBox.setModel(model.get());
    listBox.setMultipleSelectionEnabled(false);
    addAndMakeVisible(listBox);

    // Setup search box
    searchBox.setTextToShowWhenEmpty("Search annotations...", juce::Colours::grey);
    searchBox.setMultiLine(false);
    searchBox.setReturnKeyStartsNewLine(false);
    searchBox.setScrollbarsShown(false);
    searchBox.setCaretVisible(true);
    searchBox.setPopupMenuEnabled(true);
    searchBox.addListener(this);
    addAndMakeVisible(searchBox);

    // Selected annotation alpha slider
    selectedAlphaSlider.setRange(0.0, 1.0, 0.01);
    selectedAlphaSlider.setValue(0.5);
    selectedAlphaSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    selectedAlphaSlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    addAndMakeVisible(selectedAlphaSlider);

    selectedAlphaLabel.setText("Annotation Visibility", juce::dontSendNotification);
    selectedAlphaLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(selectedAlphaLabel);

    // Unselected annotation alpha slider
    pathHistorySlider.setRange(0.0, 1.0, 0.01);
    pathHistorySlider.setValue(0.0);
    pathHistorySlider.setSliderStyle(juce::Slider::LinearHorizontal);
    pathHistorySlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    addAndMakeVisible(pathHistorySlider);

    pathHistoryLabel.setText("Path History", juce::dontSendNotification);
    pathHistoryLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(pathHistoryLabel);

    distancePerFrameLabel.setFont(juce::Font(11.0f));
    distancePerFrameLabel.setJustificationType(juce::Justification::centredRight);
    distancePerFrameLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(distancePerFrameLabel);

    pathSectionSlider.setRange(0.0, 1.0, 0.001);
    pathSectionSlider.setValue(0.0);
    pathSectionSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    pathSectionSlider.setTextBoxStyle(juce::Slider::NoTextBox, true, 0, 0);
    addAndMakeVisible(pathSectionSlider);

    pathSectionLabel.setText("Path Section", juce::dontSendNotification);
    pathSectionLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(pathSectionLabel);

    pathSectionFramesEditor.setFont(juce::Font(11.0f));
    pathSectionFramesEditor.setInputRestrictions(8, "0123456789.");
    pathSectionFramesEditor.setText("0", false);
    addAndMakeVisible(pathSectionFramesEditor);

    pathSectionDistLabel.setFont(juce::Font(11.0f));
    pathSectionDistLabel.setJustificationType(juce::Justification::centredRight);
    pathSectionDistLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(pathSectionDistLabel);

    initAlphaSliders();
    
    
    for (int i = 0; i < iconNames.size(); i++) {
      if(iconNames[i] == "polygon.svg"){
        model->polygonIcon = iconsFromZipFile[i];
      }
      
      if(iconNames[i] == "measure.svg"){
        model->measureIcon = iconsFromZipFile[i];
      }
      
      if(iconNames[i] == "segment.svg"){
        model->segmentIcon = iconsFromZipFile[i];
      }

      if(iconNames[i] == "trash.svg"){
        model->trashIcon = iconsFromZipFile[i];
      }
      
    }
    
  }

  void updateAnnotations(std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > _annotations) {
    annotations = _annotations;
    filterAnnotations(searchBox.getText());
    listBox.updateContent();
    listBox.repaint();
  }

  void requestListRefresh() {
    triggerAsyncUpdate();
  }
  
  void updatelist(){
    filterAnnotations(searchBox.getText());
  }

  void newSelection();

  void filterAnnotations(const juce::String& searchText) {
    auto sortByName = [](const std::shared_ptr<Annotation>& a, const std::shared_ptr<Annotation>& b) {
      return a->getName().compareIgnoreCase(b->getName()) < 0;
    };

    if (searchText.isEmpty()) {
      filteredAnnotations = annotations;
    } else {
      auto newFiltered = std::make_shared<std::vector<std::shared_ptr<Annotation>>>();
      for (const auto& anno : *annotations) {
        if (anno->getName().containsIgnoreCase(searchText)) {
          newFiltered->push_back(anno);
        }
      }
      filteredAnnotations = newFiltered;
    }
    std::sort(filteredAnnotations->begin(), filteredAnnotations->end(), sortByName);
    model->annotations = filteredAnnotations;
    listBox.updateContent();
    listBox.repaint();
  }

  void textEditorTextChanged(juce::TextEditor& editor) override {
    filterAnnotations(editor.getText());
  }

  void setSearchText(const juce::String& text) {
    searchBox.setText(text);
    filterAnnotations(text);
  }

  void paint(Graphics &g) override {  }

  void setDistancePerFrame(std::optional<double> val, size_t frameCount = 0) {
    if (val.has_value())
      distancePerFrameLabel.setText(juce::String((int) frameCount) + " frames  |  " + juce::String(val.value(), 2) + " px/frame", juce::dontSendNotification);
    else
      distancePerFrameLabel.setText("", juce::dontSendNotification);
  }

  void setPathSectionDist(std::optional<double> val) {
    if (val.has_value())
      pathSectionDistLabel.setText(juce::String(val.value(), 2) + " px/frame", juce::dontSendNotification);
    else
      pathSectionDistLabel.setText("", juce::dontSendNotification);
  }

  void resized() override {
    auto b = getLocalBounds().reduced(10);
    auto searchArea = b.removeFromTop(30);
    searchBox.setBounds(searchArea);
    b.removeFromTop(5);

    auto pathSectionDistArea = b.removeFromBottom(14);
    pathSectionDistLabel.setBounds(pathSectionDistArea);
    auto pathSectionLabelArea = b.removeFromBottom(18);
    pathSectionLabel.setBounds(pathSectionLabelArea);
    auto pathSectionRow = b.removeFromBottom(24);
    auto pathSectionEditorArea = pathSectionRow.removeFromRight(52);
    pathSectionFramesEditor.setBounds(pathSectionEditorArea);
    pathSectionSlider.setBounds(pathSectionRow);

    auto distanceLabelArea = b.removeFromBottom(14);
    distancePerFrameLabel.setBounds(distanceLabelArea);
    auto unselectedLabelArea = b.removeFromBottom(18);
    pathHistoryLabel.setBounds(unselectedLabelArea);
    auto unselectedSliderArea = b.removeFromBottom(24);
    pathHistorySlider.setBounds(unselectedSliderArea);

    auto selectedLabelArea = b.removeFromBottom(18);
    selectedAlphaLabel.setBounds(selectedLabelArea);
    auto selectedSliderArea = b.removeFromBottom(24);
    selectedAlphaSlider.setBounds(selectedSliderArea);

    b.removeFromBottom(5);
    listBox.setBounds(b);
  }
  
private:
  void handleAsyncUpdate() override
  {
    updatelist();           // runs on message thread
  }

  void initAlphaSliders();

  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > filteredAnnotations;
  AnnoListBox listBox;
  std::unique_ptr < AnnoListBoxModel >  model;
  juce::TextEditor searchBox;

  juce::Slider selectedAlphaSlider;
  juce::Label selectedAlphaLabel;
  juce::Slider pathHistorySlider;
  juce::Label pathHistoryLabel;
  juce::Label distancePerFrameLabel;

  juce::Slider pathSectionSlider;
  juce::Label pathSectionLabel;
  juce::TextEditor pathSectionFramesEditor;
  juce::Label pathSectionDistLabel;

  AnnotateComponent * parent;

};


class NavPathListComponent : public juce::Component, private juce::ListBoxModel {
public:
  NavPathListComponent(AnnotateComponent *parent);

  void setPaths(const std::vector<NavigationPath> &newPaths);

  // Returns the currently selected path, or nullptr if none selected.
  const NavigationPath *getSelectedPath() const;

  void resized() override;
  void paint(juce::Graphics &g) override {}

private:
  int getNumRows() override { return (int) paths.size(); }

  void paintListBoxItem(int row, juce::Graphics &g, int width, int height, bool rowIsSelected) override;

  void listBoxItemClicked(int row, const juce::MouseEvent &) override;

  void backgroundClicked(const juce::MouseEvent &) override;

  AnnotateComponent *parent;
  std::vector<NavigationPath> paths;
  int selectedIndex = -1;
  juce::ListBox listBox {"nav paths", this};
  juce::Label titleLabel;
};

#endif /* AnnoListComponent_hpp */
