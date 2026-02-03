//
//  ListComponent.cpp
//  pathCam
//
//  Created by Brian Summa on 5/1/24.
//

#include "JuceHeader.h"

ListComponent::ListComponent(AnnoListBoxModel *parent) : parent(parent){
  trashButton.reset( new SvgButton ("Trash", parent->trashIcon) );
  trashButton->addListener(this);
  addChildComponent(*trashButton);
  
  colorButton.reset(new ColorButton("Color"));
  colorButton->addListener(this);
  addChildComponent(*colorButton);
  
  textEditor.reset(new TextEditor());
  addChildComponent(*textEditor);
  textEditor->setMultiLine(false);
  textEditor->setReturnKeyStartsNewLine(false);
  textEditor->setScrollbarsShown(false);
  textEditor->setCaretVisible(true);
  textEditor->setPopupMenuEnabled(true);
  textEditor->addListener(this);
}

void ListComponent::resized() {

  auto bounds = getLocalBounds().reduced(2,2);
    
  if(row_number < parent->annotations->size() ){
    auto anno = (*parent->annotations)[row_number];
    colorButton->setColor(anno->getColor());
    trashButton->setBounds(bounds.removeFromRight(bounds.getHeight()));
    colorButton->setBounds(bounds.removeFromRight(bounds.getHeight()));
    bounds.removeFromLeft(bounds.getHeight());
    bounds.reduce(4,0);
    textEditor->setText(anno->getName());
    textEditor->setBounds(bounds);
  }

}

void ListComponent::textEditorTextChanged(juce::TextEditor& editor)
{
  if(row_number >= (int)parent->annotations->size()) return;
  auto anno = (*parent->annotations)[row_number];
  anno->setName(editor.getText());
}

void ListComponent::setData(int _row_number) {
  row_number = _row_number;
}

void ListComponent::paint(Graphics &g) {
  if(row_number >= (int)parent->annotations->size()) return;
  auto anno = (*parent->annotations)[row_number];

  if (parent->parent->getSelected() == anno){
    g.fillAll(Colours::lightyellow);
  }else{
    g.fillAll(Colours::white);
  }

  auto bounds = getLocalBounds().reduced(2,2);
  
  
  if( dynamic_cast < PolygonAnnotation * > (anno.get()) != NULL){
    parent->polygonIcon->drawWithin(g, bounds.removeFromLeft(bounds.getHeight()).toFloat(),
                            juce::RectanglePlacement::centred, 1.0f);
  }
  
  if( dynamic_cast < MeasureAnnotation * > (anno.get()) != NULL){
    parent->measureIcon->drawWithin(g, bounds.removeFromLeft(bounds.getHeight()).toFloat(),
                            juce::RectanglePlacement::centred, 1.0f);
  }

  if( dynamic_cast < SegmentAnnotation * > (anno.get()) != NULL){
    parent->segmentIcon->drawWithin(g, bounds.removeFromLeft(bounds.getHeight()).toFloat(),
                            juce::RectanglePlacement::centred, 1.0f);
  }

  
  if (parent->parent->getSelected() != anno){
    colorButton->setVisible(false);
    trashButton->setVisible(false);
    textEditor->setVisible(false);
    g.setColour(anno->getColor());
    g.fillRect(bounds.removeFromRight(bounds.getHeight()));
    g.setColour(Colours::black);
    bounds.removeFromLeft(4);
    g.drawText(anno->getName(), bounds, Justification::centredLeft, true);

  }else{
    colorButton->setVisible(true);
    trashButton->setVisible(true);
    textEditor->setVisible(true);
    g.setColour(Colours::black);
    g.drawRect(colorButton->getBounds());
  }
  

  

}


void ListComponent::mouseDown(const juce::MouseEvent& event){
  if(row_number >= (int)parent->annotations->size()) return;
  parent->listBoxItemClicked(row_number, event);
  resized();
}


void ListComponent::messageBoxCallback(int result, ListComponent* caller)
{
  switch (result)
  {
    case 1:
      juce::Logger::writeToLog("User pressed Yes");
      caller->parent->removeSelected();
      break;
    default:
      break;
  }
}


void ListComponent::buttonClicked(juce::Button* button){
  if(row_number >= (int)parent->annotations->size()) return;
  if(button == colorButton.get()){
    auto colourSelector = std::make_unique<ColourSelector> (ColourSelector::showAlphaChannel
                                                            | ColourSelector::showColourAtTop
                                                            | ColourSelector::editableColour
                                                            | ColourSelector::showSliders
                                                            | ColourSelector::showColourspace);

    auto anno = (*parent->annotations)[row_number];

    colourSelector->setName ("Annotation Color");
    colourSelector->setCurrentColour (anno->getColor());
    colourSelector->addChangeListener (this);
    colourSelector->setColour (ColourSelector::backgroundColourId, Colours::transparentBlack);
    colourSelector->setSize (300, 400);

    CallOutBox::launchAsynchronously (std::move (colourSelector), getScreenBounds(), nullptr);
    
  }
  
  if(button == trashButton.get()){
    
    juce::AlertWindow::showOkCancelBox(juce::AlertWindow::WarningIcon,
                                          "Confirmation",
                                          "Are you sure you want to delete the annotation?",
                                          "Yes",
                                          "No",
                                          nullptr,
                                          juce::ModalCallbackFunction::create(messageBoxCallback, this));
    
    
  }
  
  
  
}


void ListComponent::changeListenerCallback (ChangeBroadcaster* source)
{
  if (auto* cs = dynamic_cast<ColourSelector*> (source)){
    if(row_number >= (int)parent->annotations->size()) return;
    auto anno = (*parent->annotations)[row_number];
    anno->setColor(cs->getCurrentColour());
    parent->parent->repaint();
    resized();
  }

}

