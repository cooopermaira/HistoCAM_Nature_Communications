//
//  Spinnaker.cpp
//  pathCamLib
//
//  Created by Brian Summa on 4/27/23.
//

#include "pathCam.h"

using Poco::Logger;
using Poco::Util::LayeredConfiguration;
namespace pathCam{

void CameraStream::run(){
  
    parent->camlogger.information("*** IMAGE ACQUISITION ***");

    try{
        Spinnaker::GenApi::INodeMap& sNodeMap = parent->pCam->GetTLStreamNodeMap();
        CEnumEntryPtr ptrHandlingModeEntry; 
        CEnumerationPtr ptrHandlingMode = sNodeMap.GetNode("StreamBufferHandlingMode");
        ptrHandlingModeEntry = ptrHandlingMode->GetEntryByName("OldestFirst");
        ptrHandlingMode->SetIntValue(ptrHandlingModeEntry->GetValue());
        const std::string bufferModeName = ptrHandlingMode->GetCurrentEntry()->GetDisplayName().c_str();
        cout << endl << endl << "*** Buffer Handling Mode has been set to " << bufferModeName << " ***" << endl;
        //
// Set acquisition mode to continuous
//
// *** NOTES ***
// Because the example acquires and saves 10 images, setting acquisition
// mode to continuous lets the example finish. If set to single frame
// or multiframe (at a lower number of images), the example would just
// hang. This would happen because the example has been written to
// acquire 10 images while the camera would have been programmed to
// retrieve less than that.
//
// Setting the value of an enumeration node is slightly more complicated
// than other node types. Two nodes must be retrieved: first, the
// enumeration node is retrieved from the nodemap; and second, the entry
// node is retrieved from the enumeration node. The integer value of the
// entry node is then set as the new value of the enumeration node.
//
// Notice that both the enumeration and the entry nodes are checked for
// availability and readability/writability. Enumeration nodes are
// generally readable and writable whereas their entry nodes are only
// ever readable.
//
// Retrieve enumeration node from nodemap
        Spinnaker::GenApi::INodeMap& iNodeMap = parent->pCam->GetNodeMap();
        CEnumerationPtr ptrAcquisitionMode = iNodeMap.GetNode("AcquisitionMode");
        if (!IsReadable(ptrAcquisitionMode) ||
            !IsWritable(ptrAcquisitionMode))
        {
            cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
            //throw exception
        }

        // Retrieve entry node from enumeration node
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to get or set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            //throw exception
        }
        CEnumerationPtr bufferLength = iNodeMap.GetNode("TransferQueueMaxBlockCount");

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        cout << "Acquisition mode set to continuous..." << endl;
      parent->pCam->BeginAcquisition();
      
      parent->camlogger.information("Acquiring images...");
      
      interrupt = false;
      
      unsigned long i = 1;
      
      //Will run until killed
      while (!interrupt){
        try{

          //pResultImage is on the camera
          ImagePtr pResultImage = parent->pCam->GetNextImage(1000);
          
          if (pResultImage->IsIncomplete()){
            parent->camlogger.warning(Poco::format("Image incomplete: %s",
                                           Spinnaker::Image::GetImageStatusDescription(pResultImage->GetImageStatus())));
            
          }else{
            
            const size_t width = pResultImage->GetWidth();
            const size_t height = pResultImage->GetHeight();
            parent->camlogger.information(Poco::format("Got image: %u %u", (unsigned int)width, (unsigned int)height));
            
            pathCam::Image *image = new pathCam::Image();
            image->copy_in(pResultImage->GetData());
            image->increment_smart_pointer();
            parent->sCam->pass_image(image,i);
            i+=10;
            Poco::DateTime time = Poco::DateTime();
            std::string str = Poco::DateTimeFormatter::format(Poco::DateTime(), "%Y%m%d%H%M%S%i");

            //i++;
            //std::stringstream ss;
            //ss << i;

            //ss << time.year() << time.month();
           // ss << time.day() << time.hour();
           // ss << time.minute() << time.millisecond();
            
            parent->camlogger.information(Poco::format("%s", str));

            cache_element image_in_cache;
            image_in_cache.image = image;
            image_in_cache.name = str;
            
            parent->cache_mutex.lock();
            parent->camlogger.information("Put on Queue");

            parent->cache->push(image_in_cache);
            
            parent->cache_mutex.unlock();
            
                      
          }
          
          pResultImage->Release();
          
        }
        catch (Spinnaker::Exception& e)
        {
          parent->camlogger.error("Error: %s", e.what());
        }
      }
      
      parent->pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception& e){
      parent->camlogger.error("Error: %s", e.what());
      return;
    }
        
  }

void FileStream::run(){
  
  parent->IOlogger.information("*** FILE IO ***");
  
  Poco::Thread::sleep(50);

  interrupt = false;
  
  while( !interrupt || !parent->cache->empty()){

      parent->IOlogger.information("loading next image");
    
      if (parent->thread_safe_cache_size() == 0) { Poco::Thread::sleep(100);  continue; }
    parent->cache_mutex.lock();
    cache_element front = parent->cache->front();
    parent->cache->pop();
    parent->cache_mutex.unlock();
    Image * image = front.image;
    std::string name = front.name + ".Raw";
    
    size_t image_bytes = image->width*image->height;
    Poco::Path image_path = parent->getRootPath();

    parent->caputure_set_mutex.lock();
    image_path.append(Poco::Path(parent->captureSetName));
    parent->caputure_set_mutex.unlock();
    image_path.append(Poco::Path(name));
    image->set_disk_file(image_path);
    
    //std::cout << image_path.toString() << "\n";
    /*
    Mat image_Mat = cv::Mat(Size(image->height,image->width), CV_8U, image->get_Raw(), Mat::AUTO_STEP);
    cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);
    std::string namep = front.name + ".png";
    imwrite(namep, image_Mat);
    */
    std::fstream myfile;
    myfile = std::fstream(image_path.toString(), std::ios::out | std::ios::binary);
    if (myfile.fail()) {
        std::cout << strerror(errno);
    }
    myfile.write(image->get_Raw(), image_bytes);
    image->free_memory_RAW();
    image->set_disk_file(image_path.toString());
    parent->IOlogger.information(Poco::format("Wrote: %s", image_path.toString()));
    

  }


}

void ProcessStream::run() {

    parent->sCam->spin_run();

}

SpinPath::SpinPath(LayeredConfiguration::Ptr config):  camChannel(new SimpleFileChannel), camlogger(Poco::Logger::get("CamLogger")),
                       IOChannel(new SimpleFileChannel), IOlogger(Poco::Logger::get("IOLogger")),sCam(new StreamCam(config))
{

  cache = new std::queue < cache_element >();
  camlogger.setChannel(camChannel);
  camChannel->setProperty("path", "camera.log");
  camChannel->setProperty("rotation", "2 K");
                         
  IOlogger.setChannel(IOChannel);
  IOChannel->setProperty("path", "IO.log");
  IOChannel->setProperty("rotation", "2 K");

  system = System::GetInstance();
  
  // Print out current library version
  const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
  
  
  camlogger.information(Poco::format("Spinnaker library version: %u.%u.%u.%u",
                                     spinnakerLibraryVersion.major,
                                     spinnakerLibraryVersion.minor,
                                     spinnakerLibraryVersion.type,
                                     spinnakerLibraryVersion.build));
  
  
  
  
  // Retrieve list of cameras from the system
  camList = system->GetCameras();
  
  const unsigned int numCameras = camList.GetSize();
    
  camlogger.information(Poco::format("Number of cameras detected: %u",
                                     numCameras));
  
  if (numCameras != 1){
    // Clear camera list before releasing system
    camList.Clear();
    
    // Release system
    system->ReleaseInstance();
    
    camlogger.error("No or multiple cameras deteched!");
    
  }
  
  pCam = nullptr;
  
  pCam = camList.GetByIndex(0);
  
  
}

SpinPath::~SpinPath(){
  pCam = nullptr;
  camList.Clear();
  system->ReleaseInstance();
  delete sCam;
}

//int SpinPath::Aquisition(){
//  int result = 0;
//
//  camlogger.information("*** IMAGE ACQUISITION ***");
//
//  try{
//
//
//    pCam->BeginAcquisition();
//
//    camlogger.information("Acquiring images...");
//
//    interrupt = false;
//
//    //Will run until killed
//    while (!interrupt){
//      try{
//
//        //pResultImage is on the camera
//        ImagePtr pResultImage = pCam->GetNextImage(1000);
//
//        if (pResultImage->IsIncomplete()){
//          camlogger.warning(Poco::format("Image incomplete: %s",
//                                         Spinnaker::Image::GetImageStatusDescription(pResultImage->GetImageStatus())));
//
//        }else{
//
//          const size_t width = pResultImage->GetWidth();
//          const size_t height = pResultImage->GetHeight();
//
//          pathCam::Image *image = new pathCam::Image();
//          image->copy_in(pResultImage->GetData());
//
//        }
//
//        pResultImage->Release();
//
//      }
//      catch (Spinnaker::Exception& e)
//      {
//        camlogger.error("Error: %s", e.what());
//        result = -1;
//      }
//    }
//
//    pCam->EndAcquisition();
//  }
//  catch (Spinnaker::Exception& e){
//    camlogger.error("Error: %s", e.what());
//    return -1;
//  }
//
//  return result;
//}

//int SpinPath::FileIOThread(){
//
//  camlogger.information("Starting File IO Thread");
//  while(!interrupt || !cache.empty()){
//
//    cache_mutex.lock();
//    cache_element front = cache.front();
//    cache.pop();
//    cache_mutex.unlock();
//    Image * image = front.image;
//    std::string name = front.name;
//
//    //  auto myfile = std::fstream("file.binary", std::ios::out | std::ios::binary);
//    //  myfile.write(image->get_Raw(), bytes);
//
//    //image->get_Raw()
//    //delete image;
//
//
//  }
//
//
//  camlogger.information("Stopping File IO Thread");
//
//  return 1;
//}

int SpinPath::spinUpCamera(){

  int result;
  
  try{
   
    INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
    result = PrintDeviceInfo(nodeMapTLDevice);
    
    // Initialize camera
    pCam->Init();
    
    INodeMap& nodeMap = pCam->GetNodeMap();

        
    // Configure heartbeat for GEV camera
    result = result | ResetGVCPHeartbeat();
    
    CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
    if (!IsReadable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)){
      camlogger.error("Unable to set acquisition mode to continuous (enum retrieval). Aborting...");
      return -1;
    }
    
    // Retrieve entry node from enumeration node
    CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
    if (!IsReadable(ptrAcquisitionModeContinuous)){
      camlogger.error("Unable to get or set acquisition mode to continuous (entry retrieval). Aborting...");
      return -1;
    }
    
    // Retrieve integer value from entry node
    const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();
    
    // Set integer value from entry node as new value of enumeration node
    ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);
    
    camlogger.information("Acquisition mode set to continuous...");
    
  }
  catch (Spinnaker::Exception& e){
    camlogger.error(Poco::format("Error Spinning up camera: %s", e.what()));
    result = -1;
  }
 
  return 0;
}

void SpinPath::spinDownCamera(){
  
  try{
    // Deinitialize camera
    pCam->DeInit();
  }
  catch (Spinnaker::Exception& e){
    camlogger.error(Poco::format("Error Spinning down camera: %s", e.what()));
  }
    
}

int SpinPath::ConfigureGVCPHeartbeat(bool enable){
  // Retrieve TL device nodemap
  INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
  
  // Retrieve GenICam nodemap
  INodeMap& nodeMap = pCam->GetNodeMap();
  
  CEnumerationPtr ptrDeviceType = nodeMapTLDevice.GetNode("DeviceType");
  if (!IsReadable(ptrDeviceType)){
    return -1;
  }
  
  if (ptrDeviceType->GetIntValue() != DeviceType_GigEVision){
    return 0;
  }
  
  if (enable){
    camlogger.information("Resetting heartbeat...");
  }
  else{
    camlogger.information("Disabling heartbeat...");
  }
  
  CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
  if (!IsWritable(ptrDeviceHeartbeat))
  {
    camlogger.warning("Unable to configure heartbeat. Continuing with execution as this may be non-fatal...");
  }
  else
  {
    ptrDeviceHeartbeat->SetValue(enable);
    
    if (!enable)
    {
      camlogger.warning("WARNING: Heartbeat has been disabled for the rest of this example run.");
      camlogger.warning("         Heartbeat will be reset upon the completion of this run.  If the ");
      camlogger.warning("         program is aborted unexpectedly before the heartbeat is reset, the");
      camlogger.warning("         camera may need to be power cycled to reset the heartbeat.");
    }
    else
    {
      camlogger.information("Heartbeat has been reset.");
    }
  }
  
  return 0;
}

int SpinPath::PrintDeviceInfo(INodeMap& nodeMap){
  int result = 0;
  
  camlogger.information("*** DEVICE INFORMATION ***");
  
  try{
    FeatureList_t features;
    const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
    if (IsReadable(category)){
      category->GetFeatures(features);
      
      for (auto it = features.begin(); it != features.end(); ++it){
        const CNodePtr pfeatureNode = *it;
        
        CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
        camlogger.information(Poco::format("%s : %s",
                                           pfeatureNode->GetName(),
                                           (IsReadable(pValue) ? pValue->ToString() : "Node not readable")));
      }
    }
    else{
      camlogger.warning("Device control information not available.");
    }
  }
  catch (Spinnaker::Exception& e){
    camlogger.error(Poco::format("Error: %s", e.what()));
    result = -1;
  }
  
  return result;
}


int SpinPath::startCamera(){
  
  int result = spinUpCamera();
  
  if(result != -1){
    cameraStream = new CameraStream(this);
    fileStream =  new FileStream(this);
    processStream = new ProcessStream(this);
    
    thread_cam.setOSPriority(Poco::Thread::getMaxOSPriority());
    thread_file.setOSPriority(Poco::Thread::getMaxOSPriority());
    thread_cam.start(*cameraStream);
    thread_file.start(*fileStream);
    sCam->microscopeInput = true;
    thread_sCam.start(*processStream);
    if (0 == 0) { int j = 0;  }
  }

 
  return result;
}

void SpinPath::stopCamera(){
  cameraStream->interrupt = true;
  fileStream->interrupt = true;
  
  sCam->microscopeInput = false;
  std::cout << "Collection complete, processing " << std::endl;
  thread_cam.join();
  thread_file.join();
  thread_sCam.join();
  
  delete cameraStream;
  delete fileStream;
  delete processStream;

  spinDownCamera();
}


void SpinPath::newCaptureSet(){
  
  Poco::DateTime time = Poco::DateTime();
  
  std::stringstream ss;
  ss << time.year() << time.month();
  ss << time.day() << time.hour();
  ss << time.minute() << time.millisecond();
  
  caputure_set_mutex.lock();
  captureSetName = ss.str();
  
  Poco::Path capture_path = getRootPath();
  capture_path.append(Poco::Path(captureSetName));
  
  Poco::File tmpDir(capture_path);
  tmpDir.createDirectories();
  
  caputure_set_mutex.unlock();

  
}

}
