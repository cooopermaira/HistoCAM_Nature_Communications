//
//  DiskStreamer.cpp
//  pathCamLib
//
//  Created by cooper maira on 12/21/23.
//

#include "pathCam.h"

namespace pathCam{

DiskStreamer::DiskStreamer(StreamCam *parent): parent(parent), successful(false){};

void DiskStreamer::run() {
    parent->microscope_input = true;

    std::ifstream infile(parent->input_images.toString().c_str());
    std::string imageFile;
    while (infile >> imageFile) {
        if (imageFile.size() == 0) { continue; }
        std::ifstream stream;
        stream.open(imageFile, std::ios::binary);

        char* raw_image_data = new char[6464 * 4852];
        stream.read(raw_image_data, 6464 * 4852);

        Image* image = new Image();
        image->copy_in(raw_image_data);
        image->set_disk_file(imageFile);
        delete raw_image_data;

        parent->pass_image(image);
    }
    parent->microscope_input = false;
    successful = true;
    std::cout << "disk empty ";
}
/*
void DiskStreamer::run1(){
  std::ifstream infile(parent->input_images.toString().c_str());
  std::string imageFile;
  int j = 0;
  while (infile >> imageFile){
    if(imageFile.size() == 0){ continue;}
    
    std::ifstream stream;
    stream.open(imageFile,std::ios::binary);
    
    char* raw_image_data = new char[6464*4852];
    stream.read(raw_image_data,6464*4852);
    
    parent->buffer_mutex->lock();
    //disk image is a list of file names which need to be kept with the data they were read from. When data is actually being pulled from the microscope the file names will be created here instead of read.
    parent->disk_image.push(imageFile);
    j++;
    //this buffer is not correctly mimicing the microscope. Currently it is a vector of char*
    parent->buffer.push(raw_image_data);
    parent->buffer_mutex->unlock();
    
    stream.close();
  }
  std::cout << j << std::endl;
  successful = true;
  parent->disk_empty = true; //termination condition for other processes
  std::cout << "disk empty ";
}
*/
}
