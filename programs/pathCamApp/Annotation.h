//
//  Annotation.h
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef Annotation_h
#define Annotation_h

#include "JuceHeader.h"

class Annotation {
public:
  Annotation(juce::String name): name(name){};
  
  virtual void paint(juce::Graphics& g, fPoint offset, fPoint scale=fPoint(1.0,1.0)) {};
  
  bool selected;
  juce::String name;
  fRectangle bounds;
};


class PolygonAnnotation : public Annotation{
public:
  PolygonAnnotation(juce::String name): Annotation(name) {};
  
  juce::Path path;
  
  void paint(juce::Graphics& g, fPoint offset, fPoint scale=fPoint(1.0,1.0)) override {
    juce::Path temp = path;

    temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));
    
    juce::Path::Iterator it(temp);
    
    g.setColour(juce::Colours::lightblue);

    while (it.next())
    {
        if (it.elementType == juce::Path::Iterator::lineTo ||
            it.elementType == juce::Path::Iterator::startNewSubPath)
        {
            g.fillEllipse(it.x1 - 10, it.y1 - 10, 2 * 10, 2 * 10);
        }
    }
    
    g.setColour(juce::Colours::lightblue.withAlpha(0.5f));
    g.fillPath(temp);
    
  }
  
  double getArea(){
    return 0.0; //calculatePolygonArea(vertices);
  }
  
private:
  double calculatePolygonArea(const std::vector<juce::Point<float>>& vertices) {
      unsigned int n = (unsigned int)vertices.size();
      double area = 0.0;

      // Calculate the area using the shoelace formula
      for (unsigned int i = 0; i < n; i++) {
        unsigned int j = (i + 1) % n; // Wrap around using modulo for the last point
          area += vertices[i].x * vertices[j].y;
          area -= vertices[j].x * vertices[i].y;
      }

      return std::abs(area / 2.0); // Return the absolute value of the area divided by 2
  }
  
};

class AudioAnnotation : public Annotation{
public:
  AudioAnnotation(juce::String name): Annotation(name) {};
  
 void paint(juce::Graphics& g, fPoint offset, fPoint scale=fPoint(1.0,1.0)) override {
    
    
  }
  
};

class SegmentAnnotation : public Annotation{
public:
  SegmentAnnotation(juce::String name): Annotation(name) {};
  
 void paint(juce::Graphics& g, fPoint offset,  fPoint scale=fPoint(1.0,1.0)) override {
    
    
  }
  
};


#endif /* Annotation_hpp */
