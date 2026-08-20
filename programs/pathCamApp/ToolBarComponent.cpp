//
//  ToolBarComponent.cpp
//  pathCamApp
//
//  Created by Brian Summa on 4/4/24.
//

#include "JuceHeader.h"


void ToolbarComponent::buttonClicked(juce::Button* button)
{
  ToolbarItemComponent *tButton = dynamic_cast<juce::ToolbarItemComponent*>(button);
  switch(tButton->getItemId()){
    case PCamToolbarItemFactory::home:
      parent->GuiEventHandler("home");
      break;
    case PCamToolbarItemFactory::open:
      parent->GuiEventHandler("open");
      break;
    case PCamToolbarItemFactory::save:
      std::cout << "save\n" << "\n";
      break;
    case PCamToolbarItemFactory::reload_config:
      std::cout << "reload_config\n" << "\n";
      break;
    case PCamToolbarItemFactory::settings:
    {
      // Create a simple settings panel with a toggle button
      class SettingsPanel : public juce::Component
      {
      public:
        SettingsPanel(MainComponent* mainComp) : mainComponent(mainComp)
        {
          keepFramesToggle.setButtonText("Keep Frames");
          keepFramesToggle.setToggleState(mainComp->keepFrames, juce::dontSendNotification);
          keepFramesToggle.onClick = [this]() {
            mainComponent->keepFrames = keepFramesToggle.getToggleState();
            if (mainComponent->sCam) {
              mainComponent->sCam->keepFrames = mainComponent->keepFrames;
              if (!mainComponent->keepFrames) {
                mainComponent->sCam->clear_disk_frames();
              }
            }

          };
          addAndMakeVisible(keepFramesToggle);

          setSize(200, 50);
        }

        void resized() override
        {
          auto b = getLocalBounds().reduced(10);
          keepFramesToggle.setBounds(b.removeFromTop(30));
        }

      private:
        juce::ToggleButton keepFramesToggle;
        MainComponent* mainComponent;
      };

      auto settingsPanel = std::make_unique<SettingsPanel>(parent);

      juce::CallOutBox::launchAsynchronously(
        std::move(settingsPanel),
        button->getScreenBounds(),
        nullptr
      );

      break;
    }
    case PCamToolbarItemFactory::capture:
      parent->GuiEventHandler("capture");
      break;
    case PCamToolbarItemFactory::annotate:
      parent->GuiEventHandler("annotate");
      break;
    case PCamToolbarItemFactory::pathCamIcon:
      std::cout << "pathCamIcon\n" << "\n";
      break;
    default:
      break;
  }
}
