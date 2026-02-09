//
//  Annotation.cpp
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#include "JuceHeader.h"



int Annotation::getType(){
  {
    PolygonAnnotation *cast = dynamic_cast < PointClickPoly * >(this);
    if(cast != NULL){ return _POLY; }
  }
  {
    DictateAnnotation *cast = dynamic_cast < DictateAnnotation * >(this);
    if(cast != NULL){ return _DICT; }
  }
  {
    SegmentAnnotation *cast = dynamic_cast < SegmentAnnotation * >(this);
    if(cast != NULL){ return _SEG; }
  }
  {
    MeasureAnnotation *cast = dynamic_cast < MeasureAnnotation * >(this);
    if(cast != NULL){ return _MEAS; }
  }

  return _NONE;
}

void debugAnnotation::setName(juce::String _name) {
  juce::String trimmed = _name.trim();

  int spaceIndex = trimmed.indexOfChar(' ');
  if (spaceIndex < 0)
  {
    // No space → can't parse two ints; still set the name
    Annotation::setName(trimmed);
    return;
  }

  juce::String firstStr  = trimmed.substring(0, spaceIndex).trim();
  juce::String secondStr = trimmed.substring(spaceIndex + 1).trim();

  bool ok1 = false;
  bool ok2 = false;

  int a = firstStr.getIntValue();
  int b = secondStr.getIntValue();

  // JUCE doesn't give a direct "is valid int" check,
  // so validate by round-tripping
  ok1 = (firstStr == juce::String(a));
  ok2 = (secondStr == juce::String(b));

  if (ok1 && ok2)
  {
    firstNum  = a;
    secondNum = b;

    points.clear();
    path.clear();
    area = 0.0;
    unSelect();

    auto polygon = mrImgSet->poly_annotations_from_frame_interval(firstNum,secondNum);
    for (auto & p : polygon) {
      add(fPoint(p.x,p.y));
    }
  }

  // Always update the visible name
  Annotation::setName(trimmed);
}
