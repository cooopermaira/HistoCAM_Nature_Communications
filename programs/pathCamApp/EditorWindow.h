#pragma once

#include "JuceHeader.h"

/**
 * EditorWindow
 * A simple DocumentWindow subclass that holds a TextEditor.
 */
class EditorWindow : public juce::DocumentWindow
{
public:
    EditorWindow(const juce::String& textToShow);
    ~EditorWindow() override = default;

    void closeButtonPressed() override;

private:
    juce::TextEditor textEditor;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (EditorWindow)
};
