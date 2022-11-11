//
//  common.h
//  pathCam
//
//  Created by Brian Summa on 10/3/22.
//

#ifndef common_h
#define common_h

#include <iostream>
#include "opencv2/core.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#endif

#include <chrono>
#include <fstream>
#include <string>

#include "Image.h"
#include "ImageList.h"
#include "FeatureDetector.h"
#include "Match.h"
#include "DescriptorMatcher.h"
#include "MotionEstimator.h"



#endif /* common_h */
