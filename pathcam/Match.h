//
//  Match.hpp
//  pathCam
//
//  Created by Brian Summa on 10/6/22.
//

#ifndef Match_h
#define Match_h

#include "pathCam.h"

namespace pathCam{



class Match{
public:
  Image * image_1;
  Image * image_2;
  
  std::vector<DMatch> good_matches;
  
  //image_2 from image_1
  cv::Mat H;
  double t_x, t_y;
  
  Match(Image * image_1, Image * image_2): image_1(image_1),
                                           image_2(image_2),
                                           t_x(0.0), t_y(0.0) {};
  
  Match(Match *to_invert){
    this->image_1 = to_invert->image_2;
    this->image_2 = to_invert->image_1;

    this->H = to_invert->H.inv();
    this->t_x = -to_invert->t_x;
    this->t_y = -to_invert->t_y;
    
    this->good_matches.resize(to_invert->good_matches.size());

    for(unsigned int i=0; i < this->good_matches.size(); i++){
      this->good_matches[i].trainIdx = to_invert->good_matches[i].queryIdx;
      this->good_matches[i].queryIdx = to_invert->good_matches[i].trainIdx;
      this->good_matches[i].distance = to_invert->good_matches[i].distance;
      //I am not dealing with imgIdx since this does not seem to be set in kour workflow
    }
    
  }

  ~Match(){};
  
  
};


class MatchMatrix{
public:
  std::vector < std:: vector < Match * > > match;

  MatchMatrix(){};
  
  void resize(unsigned long size=0){
    match.resize(size);
    for(unsigned int i=0; i < size; i++){
      match[i].resize(size, NULL);
    }
  };
  
  unsigned int getSize(){ return match.size(); }
  
  ~MatchMatrix(){
    for(unsigned int i=0; i < match.size(); i++){
      for(unsigned int j=0; j < match[i].size(); j++){
        if(match[i][j] != NULL){ delete match[i][j];}
      }
      match[i].clear();
    }
    match.clear();
  };
  
  bool output(std::vector < Image * > images){
    //Gephi 'CSV' format
    //;A;B;C;D;E
    //A;0;1;0;1;0
    //B;1;0;0;0;0
    //C;0;0;1;0;0
    //D;0;1;0;1;0
    //E;0;0;0;0;0
    
    std::ofstream outfile;
    outfile.open("matchGraph.csv");
    
    if(!outfile){ return false; }
    
    for(unsigned int i=0; i < images.size(); i++){
      outfile << ";" << images[i]->get_ImageFile().getBaseName();
    }
    outfile << "\n";
    
    if(match.size() == 0 || match.size() != images.size()){ return false; }
    
    for(unsigned int i=0; i < match.size(); i++){
      outfile << images[i]->get_ImageFile().getBaseName();
      for(unsigned int j=0; j < match[i].size(); j++){
        outfile << ";";
        outfile << (match[i][j] != NULL);
      }
      outfile << "\n";
    }
    
    
    
    outfile.close();
    
 
    return true;
    
  }
  
};

}



#endif /* Match_hpp */
