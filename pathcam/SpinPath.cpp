//
//  Spinnaker.cpp
//  pathCamLib
//
//  Created by Brian Summa on 4/27/23.
//

#include <sys/times.h>

#include "pathCam.h"

using Poco::Logger;
using Poco::Util::LayeredConfiguration;

namespace pathCam {
  void CameraStream::run() {
    parent->camlogger.information("*** IMAGE ACQUISITION ***");

    try {
      Spinnaker::GenApi::INodeMap &sNodeMap = parent->pCam->GetTLStreamNodeMap();
      CEnumEntryPtr ptrHandlingModeEntry;
      CEnumerationPtr ptrHandlingMode = sNodeMap.GetNode("StreamBufferHandlingMode");
      ptrHandlingModeEntry = ptrHandlingMode->GetEntryByName("OldestFirst");
      ptrHandlingMode->SetIntValue(ptrHandlingModeEntry->GetValue());
      const std::string bufferModeName = ptrHandlingMode->GetCurrentEntry()->GetDisplayName().c_str();
      cout << endl << endl << "*** Buffer Handling Mode has been set to " << bufferModeName << " ***" << endl;

      Spinnaker::GenApi::INodeMap &iNodeMap = parent->pCam->GetNodeMap();
      CEnumerationPtr ptrAcquisitionMode = iNodeMap.GetNode("AcquisitionMode");
      if (!IsReadable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)) {
        cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
        //throw exception
      }

      // Retrieve entry node from enumeration node
      CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
      if (!IsReadable(ptrAcquisitionModeContinuous)) {
        cout << "Unable to get or set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
        //throw exception
      }

      CEnumerationPtr bufferMode = sNodeMap.GetNode("StreamBufferCountMode");
      bufferMode->SetIntValue(bufferMode->GetEntryByName("Manual")->GetValue());

      CIntegerPtr bufferCount = sNodeMap.GetNode("StreamBufferCountManual");
      bufferCount->SetValue(128);

      //CEnumerationPtr bufferLength = iNodeMap.GetNode("TransferQueueMaxBlockCount");

      // Retrieve integer value from entry node
      const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

      // Set integer value from entry node as new value of enumeration node
      ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

      parent->pCam->BeginAcquisition();
      // cout << "Acquiring images..." << endl;

      parent->camlogger.information("Acquiring images...");

      interrupt = false;

      unsigned long i = 0;

      //Will run until killed
      parent->startTime = std::chrono::high_resolution_clock::now();
      while (!interrupt) {
        try {
          //pResultImage is on the camera
          ImagePtr pResultImage = parent->pCam->GetNextImage(3000);
          long timeStamp = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - parent->startTime).count();

          if (pResultImage->IsIncomplete()) {
            parent->camlogger.warning(Poco::format("Image incomplete: %s",
                                                   Spinnaker::Image::GetImageStatusDescription(
                                                     pResultImage->GetImageStatus())));
            cout << "image no good" << endl << endl;
          } else {
            //const size_t width = pResultImage->GetWidth();
            //const size_t height = pResultImage->GetHeight();
            //parent->camlogger.information(Poco::format("Got image: %u %u", (unsigned int)width, (unsigned int)height));

            pathCam::Image *image = new pathCam::Image(parent->sCam->image_width, parent->sCam->image_height,
                                                       parent->sCam->get_scope_radius());
            image->copy_in(pResultImage->GetData());
            image->increment_smart_pointer();
            image->timeStamp = timeStamp;

            std::string str;
            auto label = parent->getObjectiveLabel();
            if (!label.empty()) {
              str = std::to_string(i) + "_" + label + ".Raw";
              image->set_observed_label(label);
            } else {
              str = std::to_string(i) + ".Raw";
            }


            Poco::Path image_path = parent->getRootPath();


            image_path.append(Poco::Path(parent->captureSetName));

            image_path.append(Poco::Path(str));
            image->set_disk_file(image_path);

            {
              Poco::FastMutex::ScopedLock lock(parent->cache_mutex);
              parent->queuedFrames = true;
              parent->cache.push(image);
            }
            parent->sCam->pass_image(image, i, true);
            ++i;
            //image->free_memory_RAW();
          }
        } catch (Spinnaker::Exception &e) {
          std::cout << "error 1 " << e.what() << std::endl;
          parent->camlogger.error("Error: %s", e.what());
        }
      }
      parent->pCam->EndAcquisition();
      parent->cameraDone = true;

      Poco::Path image_path = parent->getRootPath();
      image_path.append(Poco::Path(parent->captureSetName));
      image_path.append("ts");
      image_path.setExtension(".txt");

      parent->sCam->write_image_timestamps(image_path.toString());

      std::cout << "camera stream terminated" << std::endl;
    } catch (Spinnaker::Exception &e) {
      std::cout << "error 2" << e.what() << std::endl;
      parent->camlogger.error("Error: %s", e.what());
      return;
    }
  }



  void FileStream::run(){

    int i = 0;
    try {
      while (!interrupt || !parent->cameraDone || parent->queuedFrames){
        Image* image = nullptr;

        {
          Poco::FastMutex::ScopedLock lock(parent->cache_mutex);

          if (!parent->cache.empty()) {
            image = parent->cache.front();
            parent->cache.pop();
          } else {
            parent->queuedFrames = false;
          }
        }

        if (!image) {
          Poco::Thread::sleep(50);
          continue;
        }

        try {
          retry(3, std::chrono::milliseconds(10), [&]{
            image->write_to_path(true);
          });
          if ((i++) % 100 == 0) {
            std::cout << "wrote image " << i << std::endl;
          }
        }catch (...) {
          std::cout<<"FAILURE TO WRITE IMAGE IN FileStream::run()"<<std::endl;
        }

        image->free_memory_RAW();
      }
    }catch (const std::exception& e) {
      parent->IOlogger.fatal(
        std::string("FileStream thread terminating: ") + e.what());
    }
  }

  void ProcessStream::run() {
    parent->sCam->spin_run();
  }

  SpinPath::SpinPath(LayeredConfiguration::Ptr config) : camChannel(new SimpleFileChannel),
                                                         camlogger(Poco::Logger::get("CamLogger")),
                                                         IOChannel(new SimpleFileChannel),
                                                         IOlogger(Poco::Logger::get("IOLogger")) {
    sCam = std::make_shared<StreamCam>(config);
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

    if (numCameras != 1) {
      // Clear camera list before releasing system
      camList.Clear();

      // Release system
      system->ReleaseInstance();

      camlogger.error("No or multiple cameras deteched!");
    }

    pCam = nullptr;

    pCam = camList.GetByIndex(0);

    serialStream = new SerialStream(this);
    thread_serial.setOSPriority(Poco::Thread::getMaxOSPriority());
    thread_serial.start(*serialStream);
  }

  SpinPath::~SpinPath() {
    pCam = nullptr;
    camList.Clear();
    system->ReleaseInstance();
    thread_serial.join();
    //delete sCam;
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

  int SpinPath::spinUpCamera() {
    int result;

    try {
      INodeMap &nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
      result = PrintDeviceInfo(nodeMapTLDevice);

      // Initialize camera
      pCam->Init();

      INodeMap &nodeMap = pCam->GetNodeMap();


      // Configure heartbeat for GEV camera
      result = result | ResetGVCPHeartbeat();

      CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
      if (!IsReadable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)) {
        camlogger.error("Unable to set acquisition mode to continuous (enum retrieval). Aborting...");
        return -1;
      }

      // Retrieve entry node from enumeration node
      CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
      if (!IsReadable(ptrAcquisitionModeContinuous)) {
        camlogger.error("Unable to get or set acquisition mode to continuous (entry retrieval). Aborting...");
        return -1;
      }
      // Retrieve integer value from entry node
      const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

      // Set integer value from entry node as new value of enumeration node
      ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

      camlogger.information("Acquisition mode set to continuous...");
      cout << "Acquisition mode set to continuous..." << endl;

      pCam->GevSCPSPacketSize.SetValue(9000);

      // Set exposure time to 1500 us
      // Turn off auto exposure
      pCam->ExposureAuto.SetValue(Spinnaker::ExposureAutoEnums::ExposureAuto_Off);
      //Set exposure mode to "Timed"
      pCam->ExposureMode.SetValue(Spinnaker::ExposureModeEnums::ExposureMode_Timed);
      //Set absolute value of shutter exposure time to 1500 microseconds
      pCam->ExposureTime.SetValue(1504);

      //Turn auto gain off
      pCam->GainAuto.SetValue(Spinnaker::GainAutoEnums::GainAuto_Off);
      pCam->Gain.SetValue(0);
      cout << pCam->Gain.GetValue() << endl;

      // set gamma value to 0
      pCam->GammaEnable.SetValue(false);

      pCam->BlackLevelSelector.SetValue(Spinnaker::BlackLevelSelectorEnums::BlackLevelSelector_All);
      //Set the absolute value of brightness to 1.5%.
      pCam->BlackLevel.SetValue(1.5);

      //Set auto white balance to off
      pCam->BalanceWhiteAuto.SetValue(Spinnaker::BalanceWhiteAutoEnums::BalanceWhiteAuto_Off);
      //Select red channel balance ratio and set to 1.5
      pCam->BalanceRatioSelector.SetValue(Spinnaker::BalanceRatioSelectorEnums::BalanceRatioSelector_Red);
      pCam->BalanceRatio.SetValue(1.7);
      pCam->BalanceRatioSelector.SetValue(Spinnaker::BalanceRatioSelectorEnums::BalanceRatioSelector_Blue);
      pCam->BalanceRatio.SetValue(2.5);
    } catch (Spinnaker::Exception &e) {
      camlogger.error(Poco::format("Error Spinning up camera: %s", e.what()));
      result = -1;
    }

    return 0;
  }

  void SpinPath::spinDownCamera() {
    try {
      // Deinitialize camera
      pCam->DeInit();
    } catch (Spinnaker::Exception &e) {
      camlogger.error(Poco::format("Error Spinning down camera: %s", e.what()));
    }
  }

  int SpinPath::ConfigureGVCPHeartbeat(bool enable) {
    // Retrieve TL device nodemap
    INodeMap &nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

    // Retrieve GenICam nodemap
    INodeMap &nodeMap = pCam->GetNodeMap();

    CEnumerationPtr ptrDeviceType = nodeMapTLDevice.GetNode("DeviceType");
    if (!IsReadable(ptrDeviceType)) {
      return -1;
    }

    if (ptrDeviceType->GetIntValue() != DeviceType_GigEVision) {
      return 0;
    }

    if (enable) {
      camlogger.information("Resetting heartbeat...");
    } else {
      camlogger.information("Disabling heartbeat...");
    }

    CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
    if (!IsWritable(ptrDeviceHeartbeat)) {
      camlogger.warning("Unable to configure heartbeat. Continuing with execution as this may be non-fatal...");
    } else {
      ptrDeviceHeartbeat->SetValue(enable);

      if (!enable) {
        camlogger.warning("WARNING: Heartbeat has been disabled for the rest of this example run.");
        camlogger.warning("         Heartbeat will be reset upon the completion of this run.  If the ");
        camlogger.warning("         program is aborted unexpectedly before the heartbeat is reset, the");
        camlogger.warning("         camera may need to be power cycled to reset the heartbeat.");
      } else {
        camlogger.information("Heartbeat has been reset.");
      }
    }

    return 0;
  }

  int SpinPath::PrintDeviceInfo(INodeMap &nodeMap) {
    int result = 0;

    camlogger.information("*** DEVICE INFORMATION ***");

    try {
      FeatureList_t features;
      const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
      if (IsReadable(category)) {
        category->GetFeatures(features);

        for (auto it = features.begin(); it != features.end(); ++it) {
          const CNodePtr pfeatureNode = *it;

          CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
          camlogger.information(Poco::format("%s : %s",
                                             pfeatureNode->GetName(),
                                             (IsReadable(pValue) ? pValue->ToString() : "Node not readable")));
        }
      } else {
        camlogger.warning("Device control information not available.");
      }
    } catch (Spinnaker::Exception &e) {
      camlogger.error(Poco::format("Error: %s", e.what()));
      result = -1;
    }

    return result;
  }

  static speed_t baudToSpeed(int baud) {
    switch (baud) {
      case 9600: return B9600;
      case 115200: return B115200;
      default: throw std::runtime_error("Unsupported baud");
    }
  }

  static int openSerial(const char *device, int baud) {
    int fd = ::open(device, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0)
      throw std::runtime_error(std::string("open: ") + std::strerror(errno));

    termios tty{};
    if (tcgetattr(fd, &tty) != 0) {
      ::close(fd);
      throw std::runtime_error(std::string("tcgetattr: ") + std::strerror(errno));
    }

    speed_t spd = baudToSpeed(baud);
    cfsetispeed(&tty, spd);
    cfsetospeed(&tty, spd);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_iflag = 0;
    tty.c_lflag = 0;
    tty.c_oflag = 0;

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
      ::close(fd);
      throw std::runtime_error(std::string("tcsetattr: ") + std::strerror(errno));
    }

    fcntl(fd, F_SETFL, 0); // back to blocking mode

    return fd;
  }

  void SerialStream::run() {
    try {
      parent->serial_fd = openSerial("/dev/ttyUSB0", 9600);
    } catch (...) {
      std::lock_guard<std::mutex> lk(parent->label_mu);
      parent->latest_label = "SERIAL_OPEN_ERROR";
      return;
    }
    std::cout << "serial stream active" << std::endl;

    std::string line;
    line.reserve(128);
    char buf[256];

    while (true) {
      int n = ::read(parent->serial_fd, buf, sizeof(buf));
      if (n <= 0) continue;

      for (int i = 0; i < n; ++i) {
        char c = buf[i];
        if (c == '\n') {
          if (line.size() > 0 && line[line.size() - 1] == '\r')
            line.erase(line.size() - 1);

          if (!line.empty()) {
            std::lock_guard<std::mutex> lk(parent->label_mu);
            parent->latest_label = line;
            std::cout << parent->latest_label << std::endl;
          }
          line.clear();
        } else {
          line.push_back(c);
          if (line.size() > 1024) line.clear();
        }
      }
    }
    if (parent->serial_fd >= 0) {
      ::close(parent->serial_fd);
      parent->serial_fd = -1;
    }
  }

  int SpinPath::run() {
    int result = spinUpCamera();

    if (result != -1) {
      Poco::Path root_path = Poco::Path("/home/pathcam/pcamdata/camRecord/data/");

      setRootPath(root_path);
      newCaptureSet();
      cameraStream = new CameraStream(this);
      fileStream = new FileStream(this);
      //processStream = new ProcessStream(this);

      thread_cam.setOSPriority(Poco::Thread::getMaxOSPriority());
      thread_file.setOSPriority(Poco::Thread::getMaxOSPriority());
      thread_cam.start(*cameraStream);
      thread_file.start(*fileStream);

      sCam->fileSaveFolders.push_back(Poco::Path(captureSetName));
      sCam->microscopeInput = true;

      // thread_sCam.start(*processStream);
      // thread_sCam.join();
      sCam->spin_run();
    }


    return result;
  }

  void SpinPath::stopCamera() {
    cameraStream->interrupt = true;
    fileStream->interrupt = true;

    sCam->captureTimeMS = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - startTime).count();
    sCam->microscopeInput = false;
    std::cout << "Collection complete, processing " << std::endl;
    thread_cam.join();
    thread_file.join();

    delete cameraStream;
    cameraStream = nullptr;

    delete fileStream;
    fileStream = nullptr;

    spinDownCamera();
  }


  void SpinPath::newCaptureSet() {
    Poco::DateTime time = Poco::DateTime();

    std::stringstream ss;
    ss << time.year() << time.month();
    ss << time.day() << time.hour();
    ss << time.minute() << time.millisecond();

    Poco::FastMutex::ScopedLock lock(caputure_set_mutex);
    captureSetName = ss.str();

    Poco::Path capture_path = getRootPath();
    capture_path.append(Poco::Path(captureSetName));

    Poco::File tmpDir(capture_path);
    tmpDir.createDirectories();
  }
}
