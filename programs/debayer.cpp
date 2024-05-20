//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

#include "Poco/DirectoryIterator.h"

using Poco::DirectoryIterator;

int main( int argc, char* argv[] )
{
  
  //Should probably add fancier command line parsing
  if(argc < 3){
    std::cout << "Missing input. Use:\n";
    std::cout << "debayer <path to input image> <path to output image>\n";
    return -1;
  }
  
  Path inFile = Path(argv[1]);
  Path outFile = Path(argv[2]);
  
  if(!(inFile.isDirectory() == outFile.isDirectory())){
    std::cout << "Input needs to be both directories or files.\n";
    return -1;
  }
  
  if(inFile.isDirectory()){
    
    std::cout << "Processing Directories\n";
    
    DirectoryIterator it(inFile);
    DirectoryIterator end;
    int i = 0;
    while (it != end){
      
      Path p(it.path());
      
      if(p.getExtension() == "Raw"){
        //std::cout << "read:" << p.toString() << "\n";
        
        pathCam::Image * image = new pathCam::Image();
        
        image->set_disk_file(p);
        image->load_raw_from_disk();

        if(!image->in_memory() ){
          std::cout << "Issue loading image.\n";
          return -1;
        }
        
        //image->create_reg_image(1.0,1.0,true,cv::INTER_CUBIC, false);
          cv::Size image_size(image->width, image->height);
          Mat image_Mat = cv::Mat(image_size, CV_8U, image->get_Raw(), Mat::AUTO_STEP);
          cvtColor(image_Mat, image_Mat, COLOR_BayerBG2BGR);

        Path o = outFile;
        o.append(p.getFileName());
        o.setExtension("png");
        
        //std::cout << "write:" << o.toString() << "\n";
        if(i%100==0) {
            std::cout << std::to_string(i) << std::endl;
        }
        i++;
        imwrite(o.toString(), image_Mat);
        
        delete image;
      }
      
      ++it;
    }
    
    
    
  }else{
    
    std::cout << "Processing File\n";

    pathCam::Image * image = new pathCam::Image();
    
    image->set_disk_file(inFile);
    image->load_raw_from_disk();
    
    
    if(!image->in_memory() ){
      std::cout << "Issue loading image.\n";
      return -1;
    }
    
    image->create_reg_image(1.0,1.0,true,cv::INTER_CUBIC, false);
    
    imwrite(outFile.toString(), image->get_reg_image());

    delete image;
  }

  return 0;
  
}
