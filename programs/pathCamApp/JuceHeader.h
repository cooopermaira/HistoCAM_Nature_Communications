/*

    IMPORTANT! This file is auto-generated each time you save your
    project - if you alter its contents, your changes may be overwritten!

    This is the header file that your files should include in order to get all the
    JUCE library headers. You should avoid including the JUCE headers directly in
    your own source files, because that wouldn't pick up the correct configuration
    options for your app.

*/

#pragma once


#include <juce_core/juce_core.h>
#include <juce_data_structures/juce_data_structures.h>
#include <juce_events/juce_events.h>
#include <juce_graphics/juce_graphics.h>
#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_gui_extra/juce_gui_extra.h>
#include <juce_opengl/juce_opengl.h>

#include <math.h>

#include "pathCam.h"


using namespace juce;

typedef juce::Rectangle<float> fRectangle;
typedef juce::Rectangle<int> iRectangle;
typedef juce::Point<int> iPoint;
typedef juce::Point<float> fPoint;

template < typename T >
juce::Rectangle < T > RectCtoJ (cv::Rect_< T > r) {
 return juce::Rectangle < T >(r.x, r.y, r.width, r.height);
}

template < typename T >
cv::Rect_< T > RectJtoC (juce::Rectangle < T > r) {
 return cv::Rect_< T >(r.getX(), r.getY(), r.getWidth(), r.getHeight());
}

#include "SvgButton.h"
#include "ColorButton.h"

#include "ToolBarComponent.h"

#include "EditorWindow.h"
#include "AIOverlay.h"
#include "ReportOverlay.h"
#include "ImageViewOverlay.h"
#include "ImageViewComponent.h"

#include "CaptureOverlay.h"
#include "CaptureComponent.h"

#include "Annotation.h"
#include "ListComponent.h"
#include "AnnoListComponent.h"
#include "AnnotateOverlay.h"
#include "AnnoViewComponent.h"
#include "AnnotateComponent.h"


#include "MainComponent.h"



#if defined (JUCE_PROJUCER_VERSION) && JUCE_PROJUCER_VERSION < JUCE_VERSION
 /** If you've hit this error then the version of the Projucer that was used to generate this project is
     older than the version of the JUCE modules being included. To fix this error, re-save your project
     using the latest version of the Projucer or, if you aren't using the Projucer to manage your project,
     remove the JUCE_PROJUCER_VERSION define.
 */
 #error "This project was last saved using an outdated version of the Projucer! Re-save this project with the latest version to fix this error."
#endif

#define JUCE_APP_CONFIG_HEADER "JuceHeader.h"

#if ! JUCE_DONT_DECLARE_PROJECTINFO
namespace ProjectInfo
{
    const char* const  projectName    = "PathCam Computational Microscope";
    const char* const  companyName    = "Tulane University";
    const char* const  versionString  = "1.0.0";
    const int          versionNumber  = 0x10000;
}
#endif
