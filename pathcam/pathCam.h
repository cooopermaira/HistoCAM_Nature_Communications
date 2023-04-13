//
//  pathCam.h
//
//  Created by Brian Summa on 10/3/22.
//

#ifndef pathCam_h
#define pathCam_h

#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <queue>

#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#endif

#include "Poco/Mutex.h"
#include "Poco/Runnable.h"
#include "Poco/Thread.h"
#include "Poco/MemoryPool.h"
#include "Poco/File.h"
#include "Poco/Path.h"
#include "Poco/Environment.h"
#include "Poco/Util/XMLConfiguration.h"
#include "Poco/Util/Application.h"
#include "Poco/Util/HelpFormatter.h"

#include "util.h"
#include "Image.h"
#include "ImageList.h"
#include "FeatureDetector.h"
#include "Match.h"
#include "DescriptorMatcher.h"
#include "MotionEstimator.h"
#include "BatchCam.h"



#endif /* pathCam_h */



