//
//  Spinnaker.cpp
//  pathCamLib
//
//  Created by Brian Summa on 4/27/23.
//

#include "pathCam.h"


namespace pathCam{


SpinPath::SpinPath(){
  // Print out current library version
  const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
  cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
      << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
      << endl;

  // Retrieve list of cameras from the system
  camList = system->GetCameras();
  
  const unsigned int numCameras = camList.GetSize();

  cout << "Number of cameras detected: " << numCameras << endl << endl;

  if (numCameras != 1){
      // Clear camera list before releasing system
      camList.Clear();

      // Release system
      system->ReleaseInstance();

      cout << "No or multiple cameras deteched!" << endl;
  }
  
  pCam = nullptr;
  
  pCam = camList.GetByIndex(0);


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
        cout << endl << "Resetting heartbeat..." << endl << endl;
    }
    else{
        cout << endl << "Disabling heartbeat..." << endl << endl;
    }

    CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
    if (!IsWritable(ptrDeviceHeartbeat))
    {
        cout << "Unable to configure heartbeat. Continuing with execution as this may be non-fatal..."
            << endl
            << endl;
    }
    else
    {
        ptrDeviceHeartbeat->SetValue(enable);

        if (!enable)
        {
            cout << "WARNING: Heartbeat has been disabled for the rest of this example run." << endl;
            cout << "         Heartbeat will be reset upon the completion of this run.  If the " << endl;
            cout << "         example is aborted unexpectedly before the heartbeat is reset, the" << endl;
            cout << "         camera may need to be power cycled to reset the heartbeat." << endl << endl;
        }
        else
        {
            cout << "Heartbeat has been reset." << endl;
        }
    }

    return 0;
}

int SpinPath::PrintDeviceInfo(INodeMap& nodeMap){
    int result = 0;
    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try{
        FeatureList_t features;
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsReadable(category)){
            category->GetFeatures(features);

            for (auto it = features.begin(); it != features.end(); ++it){
                const CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                cout << endl;
            }
        }
        else{
            cout << "Device control information not available." << endl;
        }
    }
    catch (Spinnaker::Exception& e){
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int SpinPath::AcquireImages(INodeMap& nodeMap, INodeMap& nodeMapTLDevice){
    int result = 0;

    cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try{
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsReadable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode)){
            cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve entry node from enumeration node
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous)){
            cout << "Unable to get or set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        cout << "Acquisition mode set to continuous..." << endl;

        //
        // Begin acquiring images
        //
        // *** NOTES ***
        // What happens when the camera begins acquiring images depends on the
        // acquisition mode. Single frame captures only a single image, multi
        // frame captures a set number of images, and continuous captures a
        // continuous stream of images. Because the example calls for the
        // retrieval of 10 images, continuous mode has been set.
        //
        // *** LATER ***
        // Image acquisition must be ended when no more images are needed.
        //
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        //
        // Retrieve device serial number for filename
        //
        // *** NOTES ***
        // The device serial number is retrieved in order to keep cameras from
        // overwriting one another. Grabbing image IDs could also accomplish
        // this.
        //
        gcstring deviceSerialNumber("");
        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrStringSerial)){
            deviceSerialNumber = ptrStringSerial->GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        // Retrieve, convert, and save images
        const unsigned int k_numImages = 10;

        //
        // Create ImageProcessor instance for post processing images
        //
        ImageProcessor processor;

        //
        // Set default image processor color processing method
        //
        // *** NOTES ***
        // By default, if no specific color processing algorithm is set, the image
        // processor will default to NEAREST_NEIGHBOR method.
        //
        processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++){
            try{
                //
                // Retrieve next received image
                //
                // *** NOTES ***
                // Capturing an image houses images on the camera buffer. Trying
                // to capture an image that does not exist will hang the camera.
                //
                // *** LATER ***
                // Once an image from the buffer is saved and/or no longer
                // needed, the image must be released in order to keep the
                // buffer from filling up.
                //
                ImagePtr pResultImage = pCam->GetNextImage(1000);

                //
                // Ensure image completion
                //
                // *** NOTES ***
                // Images can easily be checked for completion. This should be
                // done whenever a complete image is expected or required.
                // Further, check image status for a little more insight into
                // why an image is incomplete.
                //
                if (pResultImage->IsIncomplete()){
                    // Retrieve and print the image status description
                    cout << "Image incomplete: " << Spinnaker::Image::GetImageStatusDescription(pResultImage->GetImageStatus())
                        << "..." << endl
                        << endl;
                }else{

                    const size_t width = pResultImage->GetWidth();
                    const size_t height = pResultImage->GetHeight();

                    cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;
                  
                    // Create a unique filename
                    ostringstream filename;

                    filename << "Acquisition-";
                    if (!deviceSerialNumber.empty())
                    {
                        filename << deviceSerialNumber.c_str() << "-";
                    }
                    filename << imageCnt << ".raw";

                    pResultImage->Save(filename.str().c_str(), SPINNAKER_IMAGE_FILE_FORMAT_RAW);

                    cout << "Image saved at " << filename.str() << endl;
                }

                //pResultImage is on the camera, needs to be released
                pResultImage->Release();

            }
            catch (Spinnaker::Exception& e)
            {
                cout << "Error: " << e.what() << endl;
                result = -1;
            }
        }

        //
        // End acquisition
        //
        // *** NOTES ***
        // Ending acquisition appropriately helps ensure that devices clean up
        // properly and do not need to be power-cycled to maintain integrity.
        //

        pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception& e){
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return result;
}


int SpinPath::RunCamera(){
  
    int result;

    try{
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure heartbeat for GEV camera
#ifdef _DEBUG
        result = result | DisableGVCPHeartbeat();
#else
        result = result | ResetGVCPHeartbeat();
#endif

        // Acquire images
        result = result | AcquireImages(nodeMap, nodeMapTLDevice);

#ifdef _DEBUG
        // Reset heartbeat for GEV camera
        result = result | ResetGVCPHeartbeat();
#endif

        // Deinitialize camera
        pCam->DeInit();
    }
    catch (Spinnaker::Exception& e){
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

}
