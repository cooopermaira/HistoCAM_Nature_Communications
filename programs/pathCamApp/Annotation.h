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
  enum{_NONE, _POLY, _SEG, _DICT, _MEAS};
  
  Annotation(juce::String name): name(name){
    color = colorbrewer[rand()%colorbrewer.size()];
  };
  
  juce::String getName() { return name; }
  
  juce::Colour getColor(){ return color;}
  
  virtual void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0)) {};
  
  int getType();
  
  void setName(juce::String _name){ name = _name;}
  
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
  PolygonAnnotation(juce::String name): Annotation(name), area(0.0), selected(-1) {};
  
  
  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0)) override {
    juce::Path temp = path;
    
    temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));
    
    juce::Path::Iterator it(temp);
    
    g.setColour(color.withAlpha(0.5f));
    g.fillPath(temp);
    
    if(selected){
      g.setColour(juce::Colours::greenyellow);
      
      while (it.next())
      {
        if (it.elementType == juce::Path::Iterator::lineTo ||
            it.elementType == juce::Path::Iterator::startNewSubPath)
        {
          g.fillEllipse(it.x1 - 10, it.y1 - 10, 2 * 10, 2 * 10);
        }
      }
    }
    
    if(points.size() > 2){
      std::string area = Poco::format("%.0f mm^2", getArea()*1.73*0.001);
      int text_width = g.getCurrentFont().getStringWidth(area);
      int text_height = g.getCurrentFont().getHeight();
      g.setColour (juce::Colours::black.withAlpha(0.4f));
      iRectangle textbox = iRectangle (temp.getBounds().getCentreX()-(text_width/2),
                                       temp.getBounds().getCentreY()-(text_height/2),
                                       text_width,
                                       text_height);
      g.fillRect(textbox);
      
      g.setColour (juce::Colours::white);
      g.drawFittedText(area, textbox, Justification::centred, 1);
    }
  }
  
  void rebuildPath(){
    juce::Path newPath;
    
    for(unsigned int i=0; i < points.size(); i++){
      if(i==0){
        newPath.startNewSubPath(points[i].getX(), points[i].getY());
      }else{
        newPath.lineTo(points[i].getX(), points[i].getY());
      }
    }
    path.swapWithPath(newPath);
    calculatePolygonArea();
  }
  
  bool test(fPoint clickInview, fPoint distance){
    fPoint click_distance = fPoint(20.0, 20.0)*distance;
    
    for(unsigned int i=0; i < points.size(); i++){
      fPoint p = points[i];
      float d = p.getDistanceFrom(clickInview);
      if(d < click_distance.getX() || d < click_distance.getY()){
        selected = i;
        return true;
      }
    }
    unSelect();
    return false;
  }
  
  
  void move(fPoint new_position) {
    if(selected != -1){
      points[selected] = new_position;
      rebuildPath();
    }
  }
  
  
  double getArea(){ return area; }
  
  void add(fPoint p){
    if(points.size() == 0){
      path.startNewSubPath(p.getX(), p.getY());
      points.push_back(p);
    }else if (points.size() < 3){
      path.lineTo(p.getX(), p.getY());
      points.push_back(p);
    }else{
      splitClosestEdge(p);
      rebuildPath();
    }
    calculatePolygonArea();
  }
  
  void splitClosestEdge(fPoint p){
    float min_distance = std::numeric_limits< float >::infinity();
    int min_index = 0;
    fPoint minPoint;
    
    for(unsigned int i=0; i < points.size(); i++){
      int n = (i+1)%points.size();
      Line<float> line (points[i].getX(), points[i].getY(), points[n].getX(), points[n].getY());
      fPoint pointOnLine;
      float distance = line.getDistanceFromPoint (p, pointOnLine);
      if(distance < min_distance){
        min_distance = distance;
        min_index = i;
      }
    }
    
    std::vector < fPoint > new_points;
    for(unsigned int i=0; i < points.size(); i++){
      new_points.push_back(points[i]);
      if(i==min_index){ new_points.push_back(p); }
    }
    
    points = new_points;
    
  }
  
  bool isPointSelected(){ return (selected != -1); }
  
  void unSelect(){ selected = -1; }
  
private:
  
  juce::Path path;
  std::vector < fPoint > points;
  double area;
  int selected;
  
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
  
  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0)) override {}
  
};

class SegmentAnnotation : public Annotation{
public:
  SegmentAnnotation(juce::String name): Annotation(name) {};
  
  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0)) override {}
  
  
};

class MeasureAnnotation : public Annotation{
public:
  MeasureAnnotation(juce::String name): Annotation(name), measuring(false) {};
    
  inline void setPoints(fPoint p1, fPoint p2){
    line = juce::Line(p1,p2);
  }
  
  inline void setStart(fPoint p1){
    line.setStart(p1);
  }
  
  inline void setEnd(fPoint p1){
    line.setEnd(p1);
  }
  
  inline void startMeasuring(fPoint p1){
    measuring = true;
    line.setStart(p1);
    line.setEnd(p1);
  }
  
  inline void continueMeasuring(fPoint p1){
    line.setEnd(p1);
  }
  
  inline void stopMeasuring(fPoint p1){
    measuring = false;
    line.setEnd(p1);
  }
  
  inline bool isMeasuring(){
    return measuring;
  }
  
  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0)) override {
    juce::Line <float> temp = line;
    
    temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));
    
    g.setColour(color);
    const float dashLengths[2] = { 8.0f, 8.0f };
    g.drawDashedLine(temp, dashLengths, 2, 4.0f);
    
    drawPerpendicular(g, offset, scale, line.getStart(), line.getEnd(), 30.0);
    
    float len = line.getLength();
    std::string length = Poco::format("%.0f µm", len*1.73);
    
    int text_width = g.getCurrentFont().getStringWidth(length);
    int text_height = g.getCurrentFont().getHeight();
    g.setColour (juce::Colours::black.withAlpha(0.4f));
    
    
    fPoint midpoint = temp.getPointAlongLineProportionally(0.5);
    
    iRectangle textbox = iRectangle (midpoint.getX()-(text_width/2),
                                     midpoint.getY()-(text_height/2),
                                     text_width,
                                     text_height);

    
    g.fillRect(textbox);
    
    g.setColour (juce::Colours::white);
    g.drawFittedText(length, textbox, Justification::centred, 1);

  }
  
private:
  
  juce::Line <float> line;
  bool measuring;
  
  void drawPerpendicular(juce::Graphics& g,
                                  const fPoint offset, const fPoint scale,
                                  const fPoint& p1, const fPoint& p2, double L) 
  {
    
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    double s = scale.getX();
    if(s > 0.0){
      L = L/s;
    }
    
    juce::Line < float > end1, end2;
    
    // Conditions to handle vertical slope infinity issues
    if (dy == 0) {
      end1 = juce::Line < float >(fPoint(p1.x, p1.y+(L/2)), fPoint(p1.x, p1.y-(L/2)));
      end2 = juce::Line < float >(fPoint(p2.x, p2.y+(L/2)), fPoint(p2.x, p2.y-(L/2)));
      
    } else {
      double slope = dx / dy;
      
      double dist = L / 2.0 / std::sqrt(1 + slope * slope); // half length divided by hypotenuse
      
      fPoint end1_start = fPoint(p1.x - dist, p1.y + slope * dist);
      fPoint end1_end   = fPoint(p1.x + dist, p1.y - slope * dist);
      
      fPoint end2_start = fPoint(p2.x - dist, p2.y + slope * dist);
      fPoint end2_end   = fPoint(p2.x + dist, p2.y - slope * dist);
      
      end1 = juce::Line < float >(end1_start, end1_end);
      end2 = juce::Line < float >(end2_start, end2_end);

    }
    
    end1.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    end1.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));
    end2.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    end2.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));

    g.setColour(color);
    g.drawLine(end1, 2.0f);
    g.drawLine(end2, 2.0f);
    
  }
  
  
};

#endif /* Annotation_hpp */
