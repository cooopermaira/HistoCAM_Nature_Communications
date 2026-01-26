//
//  CaptureOverlay.h
//  pathCam
//
//  Created by Brian Summa on 4/18/24.
//

#ifndef CaptureOverlay_h
#define CaptureOverlay_h

#include "JuceHeader.h"

class CaptureComponent;

class StreamCamLabelList : public juce::Component,
                           private juce::ListBoxModel
{
public:
  using OnClick = std::function<void (int index, const juce::String& label)>;

  StreamCamLabelList(std::shared_ptr<pathCam::StreamCam> sc, OnClick onClickFn)
      : sCam(sc), onClick(std::move(onClickFn))
  {
    listBox.setModel(this);
    listBox.setRowHeight(24);
    listBox.setMultipleSelectionEnabled(false);
    addAndMakeVisible(listBox);
  }

  void resized() override
  {
    listBox.setBounds(getLocalBounds());
  }

  // Call this when StreamCam label list changes.
  void refresh()
  {
    listBox.updateContent();
    listBox.repaint();
  }

private:
  // ListBoxModel
  int getNumRows() override
  {
    return sCam->get_num_slides();
  }

  void paintListBoxItem(int rowNumber, juce::Graphics& g,
                        int width, int height, bool rowIsSelected) override
  {
    if (rowIsSelected)
      g.fillAll(juce::Colours::lightblue.withAlpha(0.25f));

    if (rowNumber < 0 || rowNumber >= sCam->get_num_slides())
      return;

    g.setColour(juce::Colours::white);
    g.setFont((float)height * 0.55f);

    auto text = sCam->get_slide_label(rowNumber);
    g.drawText(text, 8, 0, width - 16, height, juce::Justification::centredLeft);
  }

  void listBoxItemClicked(int row, const juce::MouseEvent&) override
  {
    if (row < 0 || row >= sCam->get_num_slides())
      return;

    if (onClick)
      onClick(row, sCam->get_slide_label(row));
  }

  // Optional: double click behavior instead of single click
  void listBoxItemDoubleClicked(int row, const juce::MouseEvent&) override
  {
    // If you prefer double-click actions, move the callback here.
  }

private:
  std::shared_ptr<pathCam::StreamCam> sCam;
  OnClick onClick;

  juce::ListBox listBox { "streamcam labels", this };
};

class CaptureOverlay final : public Component,
                             public Button::Listener {
public:
  CaptureOverlay(CaptureComponent *parent,
                 StringArray &iconNames,
                 OwnedArray<Drawable> &iconsFromZipFile);

  ~CaptureOverlay() {
  }

  void resized() override;

private:
  void buttonClicked(juce::Button *button) override;

  std::unique_ptr<SvgButton> recordButton;
  std::unique_ptr<SvgButton> stopButton;
  std::unique_ptr<SvgButton> simulateButton;

  CaptureComponent *parent;


  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CaptureOverlay)
};


#endif /* CaptureOverlay_h */
