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
    listBox.updateContent();
  }

  void newSelection();

  void filterAnnotations(const juce::String& searchText) {
    if (searchText.isEmpty()) {
      // If search is empty, show all annotations
      filteredAnnotations = annotations;
    } else {
      // Filter annotations by name substring (case-insensitive)
      auto newFiltered = std::make_shared<std::vector<std::shared_ptr<Annotation>>>();
      for (const auto& anno : *annotations) {
        if (anno->getName().containsIgnoreCase(searchText)) {
          newFiltered->push_back(anno);
        }
      }
      filteredAnnotations = newFiltered;
    }
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


  void resized() override {
    auto b = getLocalBounds().reduced(10);
    auto searchArea = b.removeFromTop(30);
    searchBox.setBounds(searchArea);
    b.removeFromTop(5); // Add small gap
    listBox.setBounds(b);
  }
  
private:
  void handleAsyncUpdate() override
  {
    updatelist();           // runs on message thread
  }

  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > filteredAnnotations;
  AnnoListBox listBox;
  std::unique_ptr < AnnoListBoxModel >  model;
  juce::TextEditor searchBox;

  AnnotateComponent * parent;

};


#endif /* AnnoListComponent_hpp */
