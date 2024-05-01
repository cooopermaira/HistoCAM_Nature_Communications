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


class ListComponent : public juce::Component {
public:
  ListComponent(AnnoListBoxModel *parent) : parent(parent), label("", "") {
    addAndMakeVisible(label);
    label.setJustificationType(juce::Justification::centredLeft);
  }
  
  void resized() override {
    label.setBounds(getLocalBounds());
  }
  
  void setData(const juce::String& text, bool selected) {
    label.setText(text, juce::dontSendNotification);
    label.setColour(juce::Label::textColourId, selected ? juce::Colours::white : juce::Colours::black);
    label.setColour(juce::Label::backgroundColourId, selected ? juce::Colours::blue : juce::Colours::white);
  }
  
private:
  juce::Label label;
  AnnoListBoxModel *parent;
  
};



class AnnoListBoxModel : public ListBoxModel {
  friend class AnnoListComponent;
  
public:
  //std::vector < int >  items;  // This array holds the list of items
  
  AnnoListBoxModel(): edit_name(false) {
//    textEditor.setMultiLine(false);
//    textEditor.setReturnKeyStartsNewLine(false);
  };
  
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
  
private:
  juce::Drawable * polygonIcon;
  juce::Drawable * measureIcon;
  juce::Drawable * segmentIcon;
  juce::Drawable * dictateIcon;
  juce::Drawable * trashIcon;
  
  
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  AnnotateComponent *parent;
  
  bool edit_name;
  
};

class AnnoListComponent : public Component {
public:
  AnnoListComponent(AnnotateComponent *parent,
                    std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile) : listBox(AnnoListBox(parent)), parent(parent), annotations(annotations) {
    
    model.annotations = annotations;
    model.parent = parent;
    
    listBox.setModel(&model);
    listBox.setMultipleSelectionEnabled(false);
    addAndMakeVisible(listBox);
    
    
    for (int i = 0; i < iconNames.size(); i++) {
      if(iconNames[i] == "polygon.svg"){
        model.polygonIcon = iconsFromZipFile[i];
      }
      
      if(iconNames[i] == "measure.svg"){
        model.measureIcon = iconsFromZipFile[i];
      }
      
      if(iconNames[i] == "segment.svg"){
        model.segmentIcon = iconsFromZipFile[i];
      }
      
      if(iconNames[i] == "dictate.svg"){
        model.dictateIcon = iconsFromZipFile[i];
      }
      
      if(iconNames[i] == "trash.svg"){
        model.trashIcon = iconsFromZipFile[i];
      }
      
      //      if(iconNames[i] == "edit.svg"){
      //        model.editIcon = iconsFromZipFile[i];
      //      }
      //
    }
    
  }
  
  
  
  void updatelist(){
    listBox.updateContent();
  }
  
  void newSelection();
  
  void paint(Graphics &g) override {  }
  
  
  void resized() override {
    auto b = getLocalBounds().reduced(10);
    listBox.setBounds(b);
  }
  
private:
  AnnoListBox listBox;
  AnnoListBoxModel model;
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;
  
  AnnotateComponent * parent;
  
};


#endif /* AnnoListComponent_hpp */
