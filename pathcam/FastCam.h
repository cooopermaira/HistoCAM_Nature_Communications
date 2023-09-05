//
//  FastCam.h
//  pathCamLib
//
//  Created by Brian on 3/15/23.
//

#ifndef FastCam_hp
#define FastCam_hp

#include "pathCam.h"

using Poco::MemoryPool;
using Poco::Path;
using Poco::Logger;

namespace pathCam{

class FastCam: public BatchCam{
  
  friend class FeaturesRunnable;
  friend class MatchRunnable;
  
public:
  FastCam(Poco::Util::LayeredConfiguration::Ptr config);
  
  ~FastCam(){}
  
  bool run();
  
  
};

}

#endif /* FastCam_hp */
