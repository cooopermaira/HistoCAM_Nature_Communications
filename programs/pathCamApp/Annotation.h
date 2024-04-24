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
  Annotation(juce::String name): name(name){
    color = colorbrewer[rand()%colorbrewer.size()];
  };
  
  juce::String getName() { return name; }
  
  juce::Colour getColor(){ return color;}
  
  virtual void paint(juce::Graphics& g, fPoint offset, fPoint scale=fPoint(1.0,1.0)) {};
  
protected:
  juce::String name;
  juce::Colour color;
  
  std::vector < juce::Colour > colorbrewer = {
    juce::Colour( 41,211,199 ),
    juce::Colour( 255,255,179 ),
    juce::Colour( 190,186,218 ),
    juce::Colour( 251,128,114 ),
    juce::Colour( 128,177,211 ),
    juce::Colour( 253,180,98 ),
    juce::Colour( 179,222,105 ),
    juce::Colour( 217,217,217 ),
    juce::Colour( 188,128,189 ),
    juce::Colour( 204,235,197 ),
    juce::Colour( 255,237,111)
  };
};


class PolygonAnnotation : public Annotation{
public:
  PolygonAnnotation(juce::String name): Annotation(name), area(0.0) {};
  
  
  void paint(juce::Graphics& g, fPoint offset, fPoint scale=fPoint(1.0,1.0)) override {
    juce::Path temp = path;
    
    temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));
    
    juce::Path::Iterator it(temp);
    
    g.setColour(color.withAlpha(0.5f));
    g.fillPath(temp);

    g.setColour(juce::Colours::greenyellow);
    
    while (it.next())
    {
      if (it.elementType == juce::Path::Iterator::lineTo ||
          it.elementType == juce::Path::Iterator::startNewSubPath)
      {
        g.fillEllipse(it.x1 - 10, it.y1 - 10, 2 * 10, 2 * 10);
      }
    }
        
    g.setColour (juce::Colours::black.withAlpha(0.4f));
    
    if(points.size() > 2){
      iRectangle textbox = iRectangle (temp.getBounds().getCentreX()-50,
                                       temp.getBounds().getCentreY()-10,
                                       100,
                                       20);
      
      g.fillRect(textbox);
      
      g.setColour (juce::Colours::white);
      std::string area = Poco::format("%.0f mm^2", getArea()*1.73*0.001);
      g.drawFittedText(area, textbox, Justification::centred, 1);
    }
  }
  
  double getArea(){
    return area;
  }
  
  
  void add(fPoint p){
    if(points.size() == 0){
      path.startNewSubPath(p.getX(), p.getY());
    }else{
      path.lineTo(p.getX(), p.getY());
    }
    points.push_back(p);
    calculatePolygonArea();
  }
  
private:
  
  juce::Path path;
  std::vector < fPoint > points;
  double area;
  
  void calculatePolygonArea() {
    unsigned int n = (unsigned int)points.size();
    area = 0.0;
    if(n < 3){ return;}
    
    // Calculate the area using the shoelace formula
    for (unsigned int i = 0; i < n; i++) {
      unsigned int j = (i + 1) % n; // Wrap around using modulo for the last point
      area += points[i].x * points[j].y;
      area -= points[j].x * points[i].y;
    }
    
    area = std::abs(area / 2.0);
  }
  
};

class DictateAnnotation : public Annotation{
public:
  DictateAnnotation(juce::String name): Annotation(name) {};
  
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
