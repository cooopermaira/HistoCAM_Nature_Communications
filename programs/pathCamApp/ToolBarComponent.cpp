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
      std::cout << "settings\n" << "\n";
      break;
    case PCamToolbarItemFactory::capture:
      parent->GuiEventHandler("capture");
      break;
    case PCamToolbarItemFactory::annotate:
      std::cout << "annotate\n" << "\n";
      break;
    case PCamToolbarItemFactory::pathCamIcon:
      std::cout << "pathCamIcon\n" << "\n";
      break;
    default:
      break;
  }
}
