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

//==============================================================================
class ToolbarComp final : public Component
{
public:
    ToolbarComp()
    {
        // Create and add the toolbar...
        addAndMakeVisible (toolbar);

        // And use our item factory to add a set of default icons to it...
        toolbar.addDefaultItems (factory);

    }

    void resized() override
    {
        //auto toolbarThickness = (int) depthSlider.getValue();
        toolbar.setBounds (getLocalBounds().removeFromTop  (50));
    }


private:
    Toolbar toolbar;


    //==============================================================================
    class DemoToolbarItemFactory final : public ToolbarItemFactory
    {
    public:
        DemoToolbarItemFactory() {}

        //==============================================================================
        // Each type of item a toolbar can contain must be given a unique ID. These
        // are the ones we'll use in this demo.
        enum DemoToolbarItemIds
        {
            home            = 1,
            open            = 2,
            save            = 3,
            reload_config   = 4,
            settings        = 5,
            capture         = 6,
            annotate        = 7,
            juceLogoButton  = 8,
        };

        void getAllToolbarItemIds (Array<int>& ids) override
        {
            // This returns the complete list of all item IDs that are allowed to
            // go in our toolbar. Any items you might want to add must be listed here. The
            // order in which they are listed will be used by the toolbar customisation panel.

            ids.add (home);
            ids.add (open);
            ids.add (save);
            ids.add (reload_config);
            ids.add (settings);
            ids.add (capture);
            ids.add (annotate);
            ids.add (juceLogoButton);

            // If you're going to use separators, then they must also be added explicitly
            // to the list.
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
            ids.add (juceLogoButton);
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

                case juceLogoButton:
                {
                    std::stringstream ss;
                    ss <<  PROJECT_SOURCE_DIR << "/resources/pathCam.png";

                    auto drawable = std::make_unique<DrawableImage>();
                    File pathCamIconFile = File(ss.str().c_str());
                    drawable->setImage (ImageFileFormat::loadFrom(pathCamIconFile));
                    return new ToolbarButton (itemId, "PathCam", std::move (drawable), {});
                }
                default:                break;
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

    DemoToolbarItemFactory factory;
};

#endif /* ToolBarComponent_hpp */
