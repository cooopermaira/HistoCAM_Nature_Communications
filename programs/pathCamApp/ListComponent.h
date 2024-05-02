//
//  ListComponent.hpp
//  pathCam
//
//  Created by Brian Summa on 5/1/24.
//

#ifndef ListComponent_h
#define ListComponent_h

#include "JuceHeader.h"

class AnnoListBoxModel;

class ListComponent : public juce::Component, juce::TextEditor::Listener, juce::Button::Listener, juce::ChangeListener {
public:
  ListComponent(AnnoListBoxModel *parent);
  
  void resized() override;
  
  void paint(Graphics &g) override;
  
  void mouseDown(const juce::MouseEvent& event) override;
  
  void setData(int _row_number);
  
  void textEditorTextChanged(juce::TextEditor& editor) override;
  
  void buttonClicked(juce::Button* button) override;

  void changeListenerCallback (ChangeBroadcaster* source) override;

  
private:
  juce::Label label;
  AnnoListBoxModel *parent;
  
  std::unique_ptr < SvgButton > trashButton;
  std::unique_ptr < ColorButton > colorButton;
  std::unique_ptr < juce::TextEditor > textEditor;

  
  int row_number;
  
};


#endif /* ListComponent_hpp */
