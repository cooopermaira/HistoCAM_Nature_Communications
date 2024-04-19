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
  
  virtual void paint(juce::Graphics& g) {};
  
  bool selected;
  juce::String name;
};


class PolygonAnnotation : public Annotation{
public:
  PolygonAnnotation(juce::String name): Annotation(name) {};
  
  std::vector < fPoint > vertices;
  
  void paint(juce::Graphics& g) override {
    
    
  }
  
  double getArea(){ return calculatePolygonArea(vertices); }
  
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
  
 void paint(juce::Graphics& g) override {
    
    
  }
  
};

class SegmentAnnotation : public Annotation{
public:
  SegmentAnnotation(juce::String name): Annotation(name) {};
  
 void paint(juce::Graphics& g) override {
    
    
  }
  
};


#endif /* Annotation_hpp */
