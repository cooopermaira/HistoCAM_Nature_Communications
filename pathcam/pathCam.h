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
//#include <Python.h>


#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include "opencv2/cudaarithm.hpp"
#include <cuda_runtime.h>
#include <cstdio>

#include "cudaImage.h"
#include "cudaSift.h"
#include "onnxruntime_cxx_api.h"

//#include "BAM/BAM.h"

#include "cuda_bundle_adjustment_types.h"
#include "cuda_bundle_adjustment.h"

#include "eigen3/Eigen/src/Geometry/Quaternion.h"
#include "samples/object_creator.h"
#endif



#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xphoto/white_balance.hpp"
#endif

#include "Poco/Mutex.h"
#include "Poco/RWLock.h"
#include "Poco/Runnable.h"
#include "Poco/Thread.h"
#include "Poco/RunnableAdapter.h"
#include "Poco/MemoryPool.h"
#include "Poco/File.h"
#include "Poco/Path.h"
#include "Poco/Environment.h"
#include "Poco/Util/XMLConfiguration.h"
#include "Poco/Util/Application.h"
#include "Poco/ConsoleChannel.h"
#include "Poco/Util/HelpFormatter.h"
#include "Poco/SimpleFileChannel.h"
#include "Poco/FileChannel.h"
#include "Poco/LogStream.h"
#include "Poco/Process.h"
#include "Poco/DateTime.h"
#include "Poco/ThreadPool.h"
#include "Poco/DateTimeFormatter.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <torch/csrc/autograd/python_variable.h>

#include "util.h"
#include "DataObserver.h"
#include "Image.h"
#include "ImageList.h"
#include "FeatureDetector.h"
#include "Match.h"
#include "DescriptorMatcher.h"
#include "MotionEstimator.h"
#include "OverlapMatrix.h"
#include "BatchCam.h"
#include "StreamCam.h"
#include "PostProcessor.h"
#include "SIFTSearchUtils.h"



#ifdef WITH_SPINNAKER
#include "Spinnaker.h"
#include "SpinPath.h"
#endif

#include "Runnables.h"
#include "JobQueue.h"
#include "Composite.h"

#include "TiledImage.h"
#include "MRTiledImage.h"


#endif /* pathCam_h */



