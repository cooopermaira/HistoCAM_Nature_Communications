//
//  Annotation.h
//  pathCam
//
//  Created by Brian Summa on 4/19/24.
//

#ifndef Annotation_h
#define Annotation_h

#include "JuceHeader.h"

struct ConceptAttributes {
  std::optional<int> gleason_primary;
  std::optional<int> gleason_secondary;
  std::optional<double> percent_involvement;
};

struct ConceptSpan {
  std::string evidence_text;
  std::string concept_text;
  std::string concept_type;
  std::string assertion;
  ConceptAttributes attributes;

  bool slideLevel = false;

  int spanStartI = -1;
  int spanEndI = -1;
  long startMS = -1;
  long endMS = -1;
  long startFrameIdx = -1;
  long endFrameIdx = -1;
};

struct AnnotationSpan
{
  std::string label;
  std::string scope;
  int spanStartI = -1;
  int spanEndI = -1;
  long startMS = -1;
  long endMS = -1;
  long startFrameIdx = -1;
  long endFrameIdx = -1;
  std::string evidenceText;
};

class Annotation {
public:

  bool global = false;
  enum{_NONE, _POLY, _SEG, _DICT, _MEAS};

  Annotation(juce::String name): name(name){
    color = colorbrewer[rand()%colorbrewer.size()];
  };
  virtual ~Annotation() = default;

  juce::String getName() { return name; }

  juce::Colour getColor(){ return color;}

  virtual void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0), float alpha=0.5f) {};

  int getType();

  virtual void setName(juce::String _name){ name = _name;}
  void setColor(juce::Colour _color){ color = _color;}


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

  bool containsPoint(const fPoint& p) {
    rebuildPath();                 // ensure path matches points
    return path.contains(p.getX(), p.getY());
  }

  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0), float alpha=0.5f) override {
    rebuildPath();
    juce::Path temp = path;

    temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));

    juce::Path::Iterator it(temp);

    g.setColour(color.withAlpha(alpha));
      g.fillPath(temp);
    auto a = getArea();
    if(selected && points.size() > 2 && a > 0.01){
      std::string area = Poco::format("%.3f mm^2", a*pow(1.73*0.001,2));
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
    newPath.closeSubPath();
    path.swapWithPath(newPath);
    calculatePolygonArea();
  }





  double getArea(){ return area; }


protected:

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
    int k = 0;
  }

};

class PointClickPoly : public PolygonAnnotation{
public:
  PointClickPoly(juce::String name): PolygonAnnotation(name), selected(-1) {};


  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0), float alpha=0.5f) override {
    PolygonAnnotation::paint(g, offset, selected, scale, alpha);
    juce::Path temp = path;

    temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
    temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));

    juce::Path::Iterator it(temp);

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

  }

  void move(fPoint new_position) {
    if(selected != -1){
      points[selected] = new_position;
      rebuildPath();
    }
  }

  void direct_add(fPoint p) {
    points.push_back(p);
  }

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
    const float epsilon = 1e-3;  // Tolerance for comparing distances

    struct EdgeInfo {
      int index;
      float distance;
      fPoint closestPoint;
      float projection_t;  // Parameter t: 0=start, 1=end, (0,1)=on segment
    };

    std::vector<EdgeInfo> edges;
    float min_distance = std::numeric_limits<float>::infinity();

    // Gather info about all edges
    for(unsigned int i=0; i < points.size(); i++){
      int n = (i+1)%points.size();
      Line<float> line(points[i].getX(), points[i].getY(), points[n].getX(), points[n].getY());
      fPoint closestPoint;
      float distance = line.getDistanceFromPoint(p, closestPoint);

      // Calculate projection parameter t where closestPoint = A + t*(B-A)
      fPoint A = points[i];
      fPoint B = points[n];
      fPoint AB(B.getX() - A.getX(), B.getY() - A.getY());
      float ab_length_sq = AB.getX()*AB.getX() + AB.getY()*AB.getY();

      float projection_t = 0.5f;  // Default to middle
      if(ab_length_sq > epsilon) {
        fPoint AP(p.getX() - A.getX(), p.getY() - A.getY());
        projection_t = (AP.getX()*AB.getX() + AP.getY()*AB.getY()) / ab_length_sq;
      }

      edges.push_back({(int)i, distance, closestPoint, projection_t});
      min_distance = std::min(min_distance, distance);
    }

    // Find all edges tied for minimum distance
    std::vector<EdgeInfo> tied_edges;
    for(const auto& edge : edges) {
      if(edge.distance <= min_distance + epsilon) {
        tied_edges.push_back(edge);
      }
    }

    int min_index = tied_edges[0].index;

    // If multiple edges are tied, use tie-breaking logic
    if(tied_edges.size() > 1) {
      float best_score = std::numeric_limits<float>::infinity();

      for(const auto& edge : tied_edges) {
        float score = 0.0f;

        // Primary criterion: prefer edges where projection is within [0,1]
        // This means the point projects onto the actual segment, not just the extended line
        if(edge.projection_t >= 0.0f && edge.projection_t <= 1.0f) {
          // Point projects onto the segment - strongly prefer this
          score = 0.0f;
        } else if(edge.projection_t < 0.0f) {
          // Closest point is the start vertex - penalize by distance from 0
          score = 1000.0f - edge.projection_t;
        } else {
          // Closest point is the end vertex - penalize by distance from 1
          score = 1000.0f + (edge.projection_t - 1.0f);
        }

        // Secondary criterion: if still tied (both project outside or both inside),
        // prefer edge with projection closer to [0,1] range
        if(std::abs(score - best_score) < epsilon) {
          // Use projection_t proximity to [0,1] as tiebreaker
          float t_dist_to_range = 0.0f;
          if(edge.projection_t < 0.0f) {
            t_dist_to_range = -edge.projection_t;
          } else if(edge.projection_t > 1.0f) {
            t_dist_to_range = edge.projection_t - 1.0f;
          }

          score += t_dist_to_range;
        }

        if(score < best_score) {
          best_score = score;
          min_index = edge.index;
        }
      }
    }

    // Insert the new point after the chosen edge's start vertex
    std::vector<fPoint> new_points;
    for(unsigned int i=0; i < points.size(); i++){
      new_points.push_back(points[i]);
      if(i == (unsigned int)min_index){
        new_points.push_back(p);
      }
    }

    points = new_points;
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


  bool isPointSelected(){ return (selected != -1); }

  void unSelect(){ selected = -1; }

private:

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
    int k = 0;
  }

};

class VoicePointPoly : public PointClickPoly {
public:
  int spanStartI = -1;
  int spanEndI = -1;
  long startMS = -1;
  long endMS = -1;
  long startFrameIdx = -1;
  long endFrameIdx = -1;

  AnnotationSpan annospan;
  ConceptSpan conceptSpan;

  VoicePointPoly(AnnotationSpan _annospan) : PointClickPoly(_annospan.label), annospan(std::move(_annospan)) {
    spanStartI = annospan.spanStartI;
    spanEndI = annospan.spanEndI;
    startMS = annospan.startMS;
    endMS = annospan.endMS;
    startFrameIdx = annospan.startFrameIdx;
    endFrameIdx = annospan.endFrameIdx;
  };
  VoicePointPoly(ConceptSpan _conceptSpan) : PointClickPoly(_conceptSpan.concept_text), conceptSpan(_conceptSpan) {
    spanStartI = conceptSpan.spanStartI;
    spanEndI = conceptSpan.spanEndI;
    startMS = conceptSpan.startMS;
    endMS = conceptSpan.endMS;
    startFrameIdx = conceptSpan.startFrameIdx;
    endFrameIdx = conceptSpan.endFrameIdx;
  };

};

class debugAnnotation : public PointClickPoly {
public:
  debugAnnotation(juce::String name, std::shared_ptr<MRTiledImageSet> _mrImgSet): PointClickPoly(name),mrImgSet(_mrImgSet){};

  int firstNum,secondNum;
  std::shared_ptr<MRTiledImageSet> mrImgSet;

  void setName(juce::String _name) override;
};


// class DictateAnnotation : public Annotation{
// public:
//   DictateAnnotation(juce::String name): Annotation(name) {};
//
//   void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0), float alpha=0.5f) override {}
//
// };

class SegmentAnnotation : public PolygonAnnotation{
public:
  SegmentAnnotation(juce::String name): PolygonAnnotation(name) {
    fovUpperLeftCorner = {0,0};
    fovUpperLeftCorner = {0,0};
  };

  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0), float alpha=0.5f) override {
    //PolygonAnnotation::paint(g, offset, selected, scale, alpha);
    if (selected) {
      for (auto points : input) {
        juce::Point temp = juce::Point(points.x, points.y);
        temp.applyTransform(juce::AffineTransform::translation(-offset.getX(), -offset.getY()));
        temp.applyTransform(juce::AffineTransform::scale(scale.getX(), scale.getY()));
        if (points.z == 0.0) {
          g.setColour(juce::Colours::red);
        }else {
          g.setColour(juce::Colours::green);
        }
        g.fillEllipse(temp.x - 5, temp.y - 5, 10, 10);
      }
    }
  }

  void add(Point3f p) {
    input.push_back(p);
  }

  void update_FOV(fPoint _upperLeft, fPoint _lowerRight) {
    if (fovUpperLeftCorner.x == 0 && fovUpperLeftCorner.y == 0) {
      fovUpperLeftCorner.x = _upperLeft.x;
      fovUpperLeftCorner.y = _upperLeft.y;
    }
    if (fovLowerRightCorner.x == 0 && fovLowerRightCorner.y == 0) {
      fovLowerRightCorner.x = _lowerRight.x;
      fovLowerRightCorner.y = _lowerRight.y;
    }
    if (_upperLeft.x < fovUpperLeftCorner.x) {
      fovUpperLeftCorner.x = _upperLeft.x;
    }
    if (_lowerRight.x > fovLowerRightCorner.x) {
      fovLowerRightCorner.x = _lowerRight.x;
    }
    if (_upperLeft.y < fovUpperLeftCorner.y) {
      fovUpperLeftCorner.y = _upperLeft.y;
    }
    if (_lowerRight.y > fovLowerRightCorner.y) {
      fovLowerRightCorner.y = _lowerRight.y;
    }
  }

  void call_SAM() {

  }

  std::vector < Point3f > input;
  int ID;
  Point2f fovUpperLeftCorner, fovLowerRightCorner;
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

  void paint(juce::Graphics& g, fPoint offset, bool selected, fPoint scale=fPoint(1.0,1.0), float alpha=0.5f) override {
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