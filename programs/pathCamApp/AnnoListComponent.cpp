//
//  AnnoListComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"


void AnnoListBoxModel::removeSelected() {
  parent->removeSelected();
}


void AnnoListBox::mouseDown(const juce::MouseEvent &event) {
  deselectAllRows();
  ListBox::mouseDown(event);
  parent->repaint();
}


void AnnoListBoxModel::listBoxItemClicked(int row, const MouseEvent &e) {
  parent->setSelected((*annotations)[row]);
  ListBoxModel::listBoxItemClicked(row, e);
  parent->repaint();
}

void AnnoListBoxModel::listBoxItemDoubleClicked(int row, const MouseEvent &e) {
  parent->setSelected((*annotations)[row]);
  ListBoxModel::listBoxItemDoubleClicked(row, e);
  parent->repaint();
}

void AnnoListBoxModel::backgroundClicked(const MouseEvent &e) {
  parent->setSelected(NULL);
  ListBoxModel::backgroundClicked(e);
  parent->repaint();
}


void AnnoListBoxModel::paintListBoxItem(int rowNumber, Graphics &g, int width, int height, bool rowIsSelected) {
}

Component *AnnoListBoxModel::refreshComponentForRow(int rowNumber,
                                                    bool /*isRowSelected*/,
                                                    Component *existingComponentToUpdate) {
  if (!annotations) return nullptr;
  if (rowNumber < 0 || rowNumber >= (int) annotations->size()) return nullptr;

  auto *component = dynamic_cast<ListComponent *>(existingComponentToUpdate);
  if (!component)
    component = new ListComponent(this);

  component->setData(rowNumber);
  return component;
}


// Component* AnnoListBoxModel::refreshComponentForRow (int rowNumber,
//                                                      bool isRowSelected,
//                                                      Component* existingComponentToUpdate){
//
//   if(getNumRows() == 0 ){ return  nullptr; }
//   ListComponent* component = static_cast<ListComponent*>(existingComponentToUpdate);
//
//   if (component == nullptr)
//       component = new ListComponent(this);
//
//   component->setData(rowNumber);
//   component->resized();
//   return component;
// }


void AnnoListComponent::newSelection() {
  for (unsigned int i = 0; i < annotations->size(); i++) {
    if ((*annotations)[i] == parent->getSelected())
      listBox.selectRow(i);
  }
}
