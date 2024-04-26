//
//  AnnoListComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"


void AnnoListBox::mouseDown(const juce::MouseEvent& event)
{
  deselectAllRows();
  ListBox::mouseDown(event);
  parent->repaint();
}


void AnnoListBoxModel::listBoxItemClicked (int row, const MouseEvent& e){
  parent->setSelected((*annotations)[row]);
  ListBoxModel::listBoxItemClicked(row, e);
  parent->repaint();
}

void AnnoListBoxModel::backgroundClicked (const MouseEvent& e){
  parent->setSelected(NULL);
  ListBoxModel::backgroundClicked(e);
  parent->repaint();
}


void AnnoListBoxModel::paintListBoxItem(int rowNumber, Graphics& g, int width, int height, bool rowIsSelected) {
  
  auto anno = (*annotations)[rowNumber];
  
  if (parent->getSelected() == anno){
    g.fillAll(Colours::yellow);
  }else{
    g.fillAll(Colours::white);
  }
    
  auto bounds = Rectangle<int>(width, height).reduced(4,4);
  
  
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


  if (parent->getSelected() == anno){
    trashIcon->drawWithin(g, bounds.removeFromRight(height-10).toFloat(),
                          juce::RectanglePlacement::centred, 1.0f);
    editIcon->drawWithin(g, bounds.removeFromRight(height).toFloat(),
                         juce::RectanglePlacement::centred, 1.0f);
  }
  
  g.setColour((*annotations)[rowNumber]->getColor());
  auto color_rectangle = bounds.removeFromRight(height);
  g.fillRect(color_rectangle);
  
  g.setColour(Colours::black);
  g.drawRect(color_rectangle);

  g.setColour(Colours::black);
  g.drawText((*annotations)[rowNumber]->getName(), bounds, Justification::centredLeft, true);

}



void AnnoListComponent::newSelection(){
  for(unsigned int i=0; i < annotations->size(); i ++){
    if((*annotations)[i] == parent->getSelected())
      listBox.selectRow(i);
  }
}
