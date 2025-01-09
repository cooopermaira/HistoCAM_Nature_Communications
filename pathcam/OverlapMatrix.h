//
//  OverlapMatrix.h
//  pathCam
//
//  Created by Brian on 4/22/23.
//

#ifndef OverlapMatrix_h
#define OverlapMatrix_h

#include "pathCam.h"

namespace pathCam{

class OverlapMatrix{
public:
  std::vector < std:: vector < double > > overlap;
  
  OverlapMatrix(){};
  
  void resize(unsigned long size=0){
    overlap.resize(size);
    for(unsigned long i=0; i < size; i++){
      overlap[i].resize(size, 0.0);
    }
  };
  
  unsigned int getSize(){ return overlap.size(); }

  
  ~OverlapMatrix(){
    for(unsigned int i=0; i < overlap.size(); i++){
      overlap[i].clear();
    }
    overlap.clear();
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
    outfile.open("overlapGraph.csv");
    
    if(!outfile){ return false; }
    
    for(unsigned int i=0; i < images.size(); i++){
      outfile << ";" << images[i]->get_ImageFile().getBaseName();
    }
    outfile << "\n";
    
    if(overlap.size() == 0 || overlap.size() != images.size()){ return false; }
    
    for(unsigned int i=0; i < overlap.size(); i++){
      outfile << images[i]->get_ImageFile().getBaseName();
      for(unsigned int j=0; j < overlap[i].size(); j++){
        outfile << ";";
        outfile << overlap[i][j];
      }
      outfile << "\n";
    }
    
    
    
    outfile.close();
    
    
    return true;
    
  }
  
  
};


}

#endif /* OverlapMatrix_hpp */
