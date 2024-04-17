//
//  ToolBarComponent.hpp
//  pathCamApp
//
//  Created by Brian Summa on 4/4/24.
//

#ifndef ToolBarComponent_h
#define ToolBarComponent_h

#include "JuceHeader.h"

using namespace juce;

class MainComponent;

//==============================================================================
class ToolbarComponent final : public Component, public Button::Listener

{
public:
  ToolbarComponent(MainComponent *parent): parent(parent)
  {
    // Create and add the toolbar...
    addAndMakeVisible (toolbar);
    
    // And use our item factory to add a set of default icons to it...
    toolbar.addDefaultItems (factory);
    
    for(unsigned int i=0; i < toolbar.getNumItems(); i++){
      ToolbarItemComponent * toolbar_button = toolbar.getItemComponent(i);
      toolbar_button->addListener(this);
    }
    
    
  }
  
  void resized() override
  {
    toolbar.setBounds (getLocalBounds().removeFromTop  (50));
  }
  
  void buttonClicked(juce::Button* button) override;
  
private:
  Toolbar toolbar;
  MainComponent *parent;
  
  //==============================================================================
  class PCamToolbarItemFactory final : public ToolbarItemFactory
  {
  public:
    PCamToolbarItemFactory() {}
    
    //==============================================================================
    // Each type of item a toolbar can contain must be given a unique ID. These
    // are the ones we'll use in this demo.
    enum PCamToolbarItemIds
    {
      home            = 1,
      open            = 2,
      save            = 3,
      reload_config   = 4,
      settings        = 5,
      capture         = 6,
      annotate        = 7,
      pathCamIcon     = 8,
    };
    
    void getAllToolbarItemIds (Array<int>& ids) override
    {
      ids.add (home);
      ids.add (open);
      ids.add (save);
      ids.add (reload_config);
      ids.add (settings);
      ids.add (capture);
      ids.add (annotate);
      ids.add (pathCamIcon);
      ids.add (separatorBarId);
      ids.add (spacerId);
      ids.add (flexibleSpacerId);
    }
    
    void getDefaultItemSet (Array<int>& ids) override
    {
      // This returns an ordered list of the set of items that make up a
      // toolbar's default set. Not all items need to be on this list, and
      // items can appear multiple times (e.g. the separators used here).
      ids.add (home);
      ids.add (spacerId);
      ids.add (open);
      ids.add (save);
      ids.add (reload_config);
      ids.add (spacerId);
      ids.add (separatorBarId);
      ids.add (spacerId);
      ids.add (capture);
      ids.add (spacerId);
      ids.add (annotate);
      ids.add (spacerId);
      ids.add (separatorBarId);
      ids.add (flexibleSpacerId);
      ids.add (settings);
      ids.add (separatorBarId);
      ids.add (pathCamIcon);
    }
    
    ToolbarItemComponent* createItem (int itemId) override
    {
      switch (itemId)
      {
        case home:            return createButtonFromZipFileSVG (itemId, "home",     "home.svg");
        case open:            return createButtonFromZipFileSVG (itemId, "open",    "open.svg");
        case save:            return createButtonFromZipFileSVG (itemId, "save",    "save.svg");
        case reload_config:   return createButtonFromZipFileSVG (itemId, "reload_config", "reload_config.svg");
        case settings:        return createButtonFromZipFileSVG (itemId, "settings",    "settings.svg");
        case capture:         return createButtonFromZipFileSVG (itemId, "capture",     "capture.svg");
        case annotate:        return createButtonFromZipFileSVG (itemId, "annotate",   "annotate.svg");
        case pathCamIcon:     return createButtonFromZipFileSVG (itemId, "pathCamIcon",   "pathCamIcon.svg");
        default:              break;
      }
      
      return nullptr;
    }
    
  private:
    StringArray iconNames;
    OwnedArray<Drawable> iconsFromZipFile;
    
    // This is a little utility to create a button with one of the SVG images in
    // our embedded ZIP file "icons.zip"
    ToolbarButton* createButtonFromZipFileSVG (const int itemId, const juce::String& text, const juce::String& filename)
    {
      if (iconsFromZipFile.size() == 0)
      {
        //Won't work for deployment, but ok for now
        std::stringstream ss;
        ss <<  PROJECT_SOURCE_DIR << "/resources/icons.zip";
        
        // If we've not already done so, load all the images from the zip file..
        ZipFile icons (File(ss.str().c_str()));
        
        
        for (int i = 0; i < icons.getNumEntries(); ++i)
        {
          std::unique_ptr<InputStream> svgFileStream (icons.createStreamForEntry (i));
          
          if (svgFileStream.get() != nullptr)
          {
            iconNames.add (icons.getEntry (i)->filename);
            iconsFromZipFile.add (Drawable::createFromImageDataStream (*svgFileStream));
          }
        }
      }
      
      auto* image = iconsFromZipFile[iconNames.indexOf (filename)];
      return new ToolbarButton (itemId, text, image->createCopy(), {});
    }
    
  };
  
  PCamToolbarItemFactory factory;
};

#endif /* ToolBarComponent_hpp */
