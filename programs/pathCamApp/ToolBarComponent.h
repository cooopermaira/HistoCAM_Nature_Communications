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

inline File getExamplesDirectory() noexcept
{
   #ifdef PIP_JUCE_EXAMPLES_DIRECTORY
    MemoryOutputStream mo;

    auto success = Base64::convertFromBase64 (mo, JUCE_STRINGIFY (PIP_JUCE_EXAMPLES_DIRECTORY));
    ignoreUnused (success);
    jassert (success);

    return mo.toString();
   #elif defined PIP_JUCE_EXAMPLES_DIRECTORY_STRING
    return File { CharPointer_UTF8 { PIP_JUCE_EXAMPLES_DIRECTORY_STRING } };
   #else
    auto currentFile = File::getSpecialLocation (File::SpecialLocationType::currentApplicationFile);
    auto exampleDir = currentFile.getParentDirectory().getChildFile ("examples");

    if (exampleDir.exists())
        return exampleDir;

    // keep track of the number of parent directories so we don't go on endlessly
    for (int numTries = 0; numTries < 15; ++numTries)
    {
        if (currentFile.getFileName() == "examples")
            return currentFile;

        const auto sibling = currentFile.getSiblingFile ("examples");

        if (sibling.exists())
            return sibling;

        currentFile = currentFile.getParentDirectory();
    }

    return currentFile;
   #endif
}

inline std::unique_ptr<InputStream> createAssetInputStream (const char* resourcePath)
{
  #if JUCE_ANDROID
    ZipFile apkZip (File::getSpecialLocation (File::invokedExecutableFile));
    return std::unique_ptr<InputStream> (apkZip.createStreamForEntry (apkZip.getIndexOfFileName ("assets/" + String (resourcePath))));
  #else
   #if JUCE_IOS
    auto assetsDir = File::getSpecialLocation (File::currentExecutableFile)
                          .getParentDirectory().getChildFile ("Assets");
   #elif JUCE_MAC
    auto assetsDir = File::getSpecialLocation (File::currentExecutableFile)
                          .getParentDirectory().getParentDirectory().getChildFile ("Resources").getChildFile ("Assets");

    if (! assetsDir.exists())
        assetsDir = getExamplesDirectory().getChildFile ("Assets");
   #else
    auto assetsDir = getExamplesDirectory().getChildFile ("Assets");
   #endif

    auto resourceFile = assetsDir.getChildFile (resourcePath);
    jassert (resourceFile.existsAsFile());

    return resourceFile.createInputStream();
  #endif
}


inline Image getImageFromAssets (const char* assetName)
{
    auto hashCode = (juce::String (assetName) + "@juce_demo_assets").hashCode64();
    auto img = ImageCache::getFromHashCode (hashCode);

    if (img.isNull())
    {
        std::unique_ptr<InputStream> juceIconStream (createAssetInputStream (assetName));

        if (juceIconStream == nullptr)
            return {};

        img = ImageFileFormat::loadFrom (*juceIconStream);

        ImageCache::addImageToCache (img, hashCode);
    }

    return img;
}

//==============================================================================
class ToolbarDemoComp final : public Component,
                              private Slider::Listener
{
public:
    ToolbarDemoComp()
    {
        // Create and add the toolbar...
        addAndMakeVisible (toolbar);

        // And use our item factory to add a set of default icons to it...
        toolbar.addDefaultItems (factory);

        // Now we'll just create the other sliders and buttons on the demo page, which adjust
        // the toolbar's properties...
        addAndMakeVisible (infoLabel);
        infoLabel.setJustificationType (Justification::topLeft);
        infoLabel.setBounds (80, 80, 450, 100);
        infoLabel.setInterceptsMouseClicks (false, false);

        addAndMakeVisible (depthSlider);
        depthSlider.setRange (10.0, 200.0, 1.0);
        depthSlider.setValue (50, dontSendNotification);
        depthSlider.addListener (this);
        depthSlider.setBounds (80, 210, 300, 22);
        depthLabel.attachToComponent (&depthSlider, false);

        addAndMakeVisible (orientationButton);
        orientationButton.onClick = [this] { toolbar.setVertical (! toolbar.isVertical()); resized(); };
        orientationButton.changeWidthToFitText (22);
        orientationButton.setTopLeftPosition (depthSlider.getX(), depthSlider.getBottom() + 20);

        addAndMakeVisible (customiseButton);
        customiseButton.onClick = [this] { toolbar.showCustomisationDialog (factory); };
        customiseButton.changeWidthToFitText (22);
        customiseButton.setTopLeftPosition (orientationButton.getRight() + 20, orientationButton.getY());
    }

    void resized() override
    {
        auto toolbarThickness = (int) depthSlider.getValue();

        if (toolbar.isVertical())
            toolbar.setBounds (getLocalBounds().removeFromLeft (toolbarThickness));
        else
            toolbar.setBounds (getLocalBounds().removeFromTop  (toolbarThickness));
    }

    void sliderValueChanged (Slider*) override
    {
        resized();
    }

private:
    Toolbar toolbar;

    Slider depthSlider  { Slider::LinearHorizontal, Slider::TextBoxLeft };

    Label depthLabel  { {}, "Toolbar depth:" },
          infoLabel   { {}, "As well as showing off toolbars, this demo illustrates how to store "
                            "a set of SVG files in a Zip file, embed that in your application, and read "
                            "them back in at runtime.\n\nThe icon images here are taken from the open-source "
                            "Tango icon project."};

    TextButton orientationButton  { "Vertical/Horizontal" },
               customiseButton    { "Customise..." };

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
            doc_new         = 1,
            doc_open        = 2,
            doc_save        = 3,
            doc_saveAs      = 4,
            edit_copy       = 5,
            edit_cut        = 6,
            edit_paste      = 7,
            juceLogoButton  = 8,
            customComboBox  = 9
        };

        void getAllToolbarItemIds (Array<int>& ids) override
        {
            // This returns the complete list of all item IDs that are allowed to
            // go in our toolbar. Any items you might want to add must be listed here. The
            // order in which they are listed will be used by the toolbar customisation panel.

            ids.add (doc_new);
            ids.add (doc_open);
            ids.add (doc_save);
            ids.add (doc_saveAs);
            ids.add (edit_copy);
            ids.add (edit_cut);
            ids.add (edit_paste);
            ids.add (juceLogoButton);
            ids.add (customComboBox);

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
            ids.add (doc_new);
            ids.add (doc_open);
            ids.add (doc_save);
            ids.add (doc_saveAs);
            ids.add (spacerId);
            ids.add (separatorBarId);
            ids.add (edit_copy);
            ids.add (edit_cut);
            ids.add (edit_paste);
            ids.add (separatorBarId);
            ids.add (flexibleSpacerId);
            ids.add (customComboBox);
            ids.add (flexibleSpacerId);
            ids.add (separatorBarId);
            ids.add (juceLogoButton);
        }

        ToolbarItemComponent* createItem (int itemId) override
        {
            switch (itemId)
            {
                case doc_new:           return createButtonFromZipFileSVG (itemId, "new",     "document-new.svg");
                case doc_open:          return createButtonFromZipFileSVG (itemId, "open",    "document-open.svg");
                case doc_save:          return createButtonFromZipFileSVG (itemId, "save",    "document-save.svg");
                case doc_saveAs:        return createButtonFromZipFileSVG (itemId, "save as", "document-save-as.svg");
                case edit_copy:         return createButtonFromZipFileSVG (itemId, "copy",    "edit-copy.svg");
                case edit_cut:          return createButtonFromZipFileSVG (itemId, "cut",     "edit-cut.svg");
                case edit_paste:        return createButtonFromZipFileSVG (itemId, "paste",   "edit-paste.svg");

                case juceLogoButton:
                {
                    std::stringstream ss;
                    ss <<  PROJECT_SOURCE_DIR << "/resources/pathCam.png";

                    auto drawable = std::make_unique<DrawableImage>();
                    drawable->setImage (getImageFromAssets (ss.str().c_str()));
                    return new ToolbarButton (itemId, "PathCam", std::move (drawable), {});
                }

                case customComboBox:    return new CustomToolbarComboBox (itemId);
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
                // If we've not already done so, load all the images from the zip file..
                ZipFile icons (createAssetInputStream ("/Users/bsumma/Source/tulane/pathcam/build/pathCamApp_artefacts/Debug/icons.zip").release(), true);

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

        // Demonstrates how to put a custom component into a toolbar - this one contains
        // a ComboBox.
        class CustomToolbarComboBox final : public ToolbarItemComponent
        {
        public:
            CustomToolbarComboBox (const int toolbarItemId)
                : ToolbarItemComponent (toolbarItemId, "Custom Toolbar Item", false)
            {
                addAndMakeVisible (comboBox);

                for (int i = 1; i < 20; ++i)
                    comboBox.addItem ("Toolbar ComboBox item " + juce::String (i), i);

                comboBox.setSelectedId (1);
                comboBox.setEditableText (true);
            }

            bool getToolbarItemSizes (int /*toolbarDepth*/, bool isVertical,
                                      int& preferredSize, int& minSize, int& maxSize) override
            {
                if (isVertical)
                    return false;

                preferredSize = 250;
                minSize = 80;
                maxSize = 300;
                return true;
            }

            void paintButtonArea (Graphics&, int, int, bool, bool) override
            {
            }

            void contentAreaChanged (const Rectangle<int>& newArea) override
            {
                comboBox.setSize (newArea.getWidth() - 2,
                                  jmin (newArea.getHeight() - 2, 22));

                comboBox.setCentrePosition (newArea.getCentreX(), newArea.getCentreY());
            }

        private:
            ComboBox comboBox  { "demo toolbar combo box" };
        };
    };

    DemoToolbarItemFactory factory;
};

#endif /* ToolBarComponent_hpp */
