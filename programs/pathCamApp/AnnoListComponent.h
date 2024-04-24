//
//  AnnoListComponent.h
//  pathCamApp
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef AnnoListComponent_h
#define AnnoListComponent_h

#include "JuceHeader.h"


class MyListBox : public juce::ListBox
{
public:
    MyListBox()
    {
        
    }

  void mouseDown(const juce::MouseEvent& event) override
  {
    deselectAllRows();
    ListBox::mouseDown(event);
  }
};


class MyListBoxModel : public ListBoxModel {
  friend class AnnoListComponent;
public:
  std::vector < int >  items;  // This array holds the list of items
  
  int getNumRows() override {
    return (int)items.size();
  }


  void paintListBoxItem(int rowNumber, Graphics& g, int width, int height, bool rowIsSelected) override {
    if (rowIsSelected){
      g.fillAll(Colours::lightyellow);
    }else{
      g.fillAll(Colours::white);
    }
      
    auto bounds = Rectangle<int>(width, height).reduced(4,4);
    
    auto anno = (*annotations)[items[rowNumber]];
    
    if( dynamic_cast < PolygonAnnotation * > (anno.get()) != NULL){
      polygonIcon->drawWithin(g, bounds.removeFromLeft(height).toFloat(),
                              juce::RectanglePlacement::centred, 1.0f);
    }

    if( dynamic_cast < SegmentAnnotation * > (anno.get()) != NULL){
      segmentIcon->drawWithin(g, bounds.removeFromLeft(height).toFloat(),
                              juce::RectanglePlacement::centred, 1.0f);
    }

    if( dynamic_cast < DictateAnnotation * > (anno.get()) != NULL){
      dictateIcon->drawWithin(g, bounds.removeFromLeft(height).toFloat(),
                              juce::RectanglePlacement::centred, 1.0f);
    }


    if (rowIsSelected){
      trashIcon->drawWithin(g, bounds.removeFromRight(height-10).toFloat(),
                            juce::RectanglePlacement::centred, 1.0f);
      editIcon->drawWithin(g, bounds.removeFromRight(height).toFloat(),
                           juce::RectanglePlacement::centred, 1.0f);
    }
    
    g.setColour((*annotations)[items[rowNumber]]->getColor());
    auto color_rectangle = bounds.removeFromRight(height);
    g.fillRect(color_rectangle);
    
    g.setColour(Colours::black);
    g.drawRect(color_rectangle);

    g.setColour(Colours::black);
    g.drawText((*annotations)[items[rowNumber]]->getName(), bounds, Justification::centredLeft, true);

  }
  
private:
  juce::Drawable * polygonIcon;
  juce::Drawable * segmentIcon;
  juce::Drawable * dictateIcon;
  juce::Drawable * editIcon;
  juce::Drawable * trashIcon;

  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;

  
};


class AnnoListComponent : public Component {
public:
  AnnoListComponent(std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations,
                    StringArray &iconNames,
                    OwnedArray<Drawable> &iconsFromZipFile) : annotations(annotations) {
    
    listBox.setModel(&model);
    listBox.setMultipleSelectionEnabled(false);
    addAndMakeVisible(listBox);
    
    model.annotations = annotations;
    
    for (int i = 0; i < iconNames.size(); i++) {
      if(iconNames[i] == "polygon.svg"){
        model.polygonIcon = iconsFromZipFile[i];
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
      
      if(iconNames[i] == "edit.svg"){
        model.editIcon = iconsFromZipFile[i];
      }
      
    }
      
  }
  
  void updatelist(){
    model.items.clear();
    for(unsigned int i=0; i < annotations->size(); i++){
      model.items.push_back(i);
    }
    listBox.updateContent();
  }
  
  void paint(Graphics &g) override {  }
  
  
  void resized() override {
    auto b = getLocalBounds().reduced(10);
    listBox.setBounds(b);
  }
  
private:
  MyListBox listBox;
  MyListBoxModel model;
  std::shared_ptr< std::vector < std::shared_ptr<  Annotation > > > annotations;


};
#endif /* AnnoListComponent_hpp */
