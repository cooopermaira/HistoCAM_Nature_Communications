#include "JuceHeader.h"

EditorWindow::EditorWindow(const juce::String& textToShow)
    : DocumentWindow("Text Editor",
                     juce::Colours::lightgrey,
                     DocumentWindow::allButtons)
{
    setUsingNativeTitleBar(true);

    // Initialize the text editor
    textEditor.setMultiLine(true);
    textEditor.setReturnKeyStartsNewLine(true);
    textEditor.setText(textToShow, juce::dontSendNotification);

    // Add and make visible
    setContentOwned(&textEditor, true);

    // Set the size of the window
    setResizable(true, false);
    centreWithSize (400, 300);

    // Make it visible
    setVisible(true);
}

void EditorWindow::closeButtonPressed()
{
    // This is called when the user presses the window's close button.
    // Here, we'll just delete this window.
    delete this;
}
