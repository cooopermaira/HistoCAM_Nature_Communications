//
//  AnnoListComponent.h
//  pathCamApp
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef AnnoListComponent_h
#define AnnoListComponent_h

#include "JuceHeader.h"

class DraggableListBoxModel : public ListBoxModel {
public:
  StringArray items;  // This array holds the list of items
  
  int getNumRows() override {
    return items.size();
  }

  void paintListBoxItem(int rowNumber, Graphics& g, int width, int height, bool rowIsSelected) override {
    g.fillAll(Colours::white);
    if (rowIsSelected)
      g.fillAll(Colours::lightblue);
    g.setColour(Colours::black);
    g.drawText(items[rowNumber], Rectangle<int>(width, height).reduced(4, 0), Justification::centredLeft, true);
  }
  
  var getDragSourceDescription(const SparseSet<int>& selectedRows) override {
    // This tag is used for the drag-and-drop description
    return "DraggableItem";
  }
  
  //    void listBoxItemDropped(int row, const DragAndDropTarget::SourceDetails& dragSourceDetails, int insertIndex) override {
  //        auto dragDescription = dragSourceDetails.description.toString();
  //        if (dragDescription == "DraggableItem" && row != insertIndex) {
  //            juce::String item = items[row];
  //            items.remove(row);
  //            items.insert(insertIndex, item);
  //            updateContent();
  //        }
  //    }
};


class AnnoListComponent : public DragAndDropContainer, public Component {
public:
  AnnoListComponent(std::shared_ptr< std::vector < Annotation > > annotations) : annotations(annotations) {
    model.items = { "Annotation 1", "Annotation 2", "Annotation 3", "Annotation 4", "Annotation 5" };
    listBox.setModel(&model);
    listBox.setMultipleSelectionEnabled(false);
    addAndMakeVisible(listBox);
  }
  
  
  void paint(Graphics &g) override {
    g.fillAll(juce::Colours::white);  // Set the background color here
  }
  
  
  void resized() override {
    auto b = getLocalBounds().reduced(10);
    listBox.setBounds(b);
  }
  
private:
  ListBox listBox;
  DraggableListBoxModel model;
  std::shared_ptr< std::vector < Annotation > > annotations;

};
#endif /* AnnoListComponent_hpp */
