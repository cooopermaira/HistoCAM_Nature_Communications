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
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>


#include "opencv2/core.hpp"

#ifdef PATHCAM_HAS_CUDA
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
std::cerr<<"CUDA error "<<cudaGetErrorString(e)<<" @ "<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} } while(0)
#endif

#ifdef PATHCAM_OPENCV_CUDA


#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"

#endif

#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/camera.hpp>


#include <cstdio>
#include <condition_variable>
#include <thread>

#include <clipper2/clipper.h>





#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xphoto/white_balance.hpp"
#

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

#ifdef PATHCAM_HAS_TENSORRT
#include <NvInfer.h>
#endif

#include "util.h"
#include "DataObserver.h"
#include "Image.h"
#include "FeatureDetector.h"
#include "Match.h"
#include "DescriptorMatcher.h"
#include "MotionEstimator.h"
#include "OverlapMatrix.h"
#include "BatchCam.h"

#include "StreamCam.h"
#include "SIFTSearchUtils.h"


#ifdef PATHCAM_HAS_TENSORRT
#include "AccessSAM.h"
#endif

#ifdef WITH_SPINNAKER
#include "Spinnaker.h"
#include "SpinPath.h"
#endif

#ifdef PATHCAM_OPENCV_CUDA
#include "CompositeVoronoi.h"
#endif

#include "Runnables.h"
#include "JobQueue.h"
#include "Composite.h"
#include "MetricComposite.h"


#include "TiledImage.h"
#include "MRTiledImage.h"




#endif /* pathCam_h */



