//
//  AnnoListComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"
std::string to_string_sig(double x, int n) {
  std::ostringstream oss;
  oss << std::setprecision(n) << x;
  return oss.str();
}

std::string to_string_fixed(double x, int n) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(n) << x;
  return oss.str();
}

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
  component->resized();
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


void AnnoListComponent::initAlphaSliders() {
  selectedAlphaSlider.onValueChange = [this]() {
    parent->annotationVisibility = (float) selectedAlphaSlider.getValue();
    parent->rightComponent->repaint();
  };
  pathHistorySlider.onValueChange = [this]() {
    parent->pathHistory = (float) pathHistorySlider.getValue();
    parent->updateEphemeralNavPath();
  };
  pathSectionSlider.onValueChange = [this]() {
    parent->pathSection = (float) pathSectionSlider.getValue();
    parent->updatePathSection();
  };
  auto updateSectionFrames = [this]() {
    parent->pathSectionFrames = pathSectionFramesEditor.getText().getFloatValue();
    parent->updatePathSection();
  };
  pathSectionFramesEditor.onReturnKey = updateSectionFrames;
  pathSectionFramesEditor.onFocusLost = updateSectionFrames;
}

void AnnoListComponent::newSelection() {
  for (unsigned int i = 0; i < filteredAnnotations->size(); i++) {
    if ((*filteredAnnotations)[i] == parent->getSelected())
      listBox.selectRow(i);
  }
}

// ── NavPathListComponent ──────────────────────────────────────────────────────

NavPathListComponent::NavPathListComponent(AnnotateComponent *parent) : parent(parent) {
  titleLabel.setText("Navigation Paths", juce::dontSendNotification);
  titleLabel.setFont(juce::Font(12.0f, juce::Font::bold));
  titleLabel.setJustificationType(juce::Justification::centredLeft);
  addAndMakeVisible(titleLabel);

  listBox.setRowHeight(22);
  listBox.setMultipleSelectionEnabled(false);
  addAndMakeVisible(listBox);
}

void NavPathListComponent::setPaths(const std::vector<NavigationPath> &newPaths) {
  paths = newPaths;
  selectedIndex = -1;
  listBox.updateContent();
  listBox.repaint();
}

const NavigationPath *NavPathListComponent::getSelectedPath() const {
  if (selectedIndex >= 0 && selectedIndex < (int) paths.size())
    return &paths[selectedIndex];
  return nullptr;
}

void NavPathListComponent::resized() {
  auto b = getLocalBounds().reduced(10);
  titleLabel.setBounds(b.removeFromTop(18));
  listBox.setBounds(b);
}

void NavPathListComponent::paintListBoxItem(int row, juce::Graphics &g,
                                            int width, int height, bool rowIsSelected) {
  if (row < 0 || row >= (int) paths.size()) return;

  if (rowIsSelected) {
    g.fillAll(juce::Colours::lightblue.withAlpha(0.4f));
  }

  const auto &path = paths[row];
  std::string label = pathCam::Image::get_label(path.magLabel) + "  |  " + std::to_string((int) path.frameCenters.size())
  + " frames"+ "  |  " + to_string_fixed(path.distancePerFrame, 2) +" px/frame";

  g.setColour(juce::Colours::white);
  g.setFont(11.0f);
  g.drawText(label, 6, 0, width - 8, height, juce::Justification::centredLeft);
}

void NavPathListComponent::listBoxItemClicked(int row, const juce::MouseEvent &) {
  selectedIndex = row;
  parent->rightComponent->repaint();
}

void NavPathListComponent::backgroundClicked(const juce::MouseEvent &) {
  selectedIndex = -1;
  listBox.deselectAllRows();
  parent->rightComponent->repaint();
}
