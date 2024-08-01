//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"

using Poco::DirectoryIterator;

int main(int argc, char *argv[]) {

  //Should probably add fancier command line parsing
  if (argc < 3) {
    std::cout << "Missing input. Use:\n";
    std::cout << "debayer <path to input image> <path to output image>\n";
    return -1;
  }

  auto inFile = Poco::Path(argv[1]);
  auto outFile = Poco::Path(argv[2]);

  if (!(inFile.isDirectory() == outFile.isDirectory())) {
    std::cout << "Input needs to be both directories or files.\n";
    return -1;
  }

  if (inFile.isDirectory()) {
    std::vector<double> *blur = new std::vector<double>;
    std::vector<std::string>* names = new std::vector<std::string>;

    blur->resize(3000);
    names->resize(3000);

    std::cout << "Processing Directories\n";
    auto jq = pathCam::JobQueue(20, 20);

    Poco::DirectoryIterator it(inFile);
    Poco::DirectoryIterator end;
    int num = 0;
    while (it != end) {

      Poco::Path p(it.path());

      if (p.getExtension() == "Raw") {
        //std::cout << "read:" << p.toString() << "\n";

        auto *image = new pathCam::Image(6464,4852,2190);

        image->set_disk_file(p);
        auto *dr = new pathCam::DebayerRunnable(image, outFile,blur,names,num);
        jq.add_runnable(dr,num);

      }
      num++;
      ++it;
    }
    while (!jq.is_empty()) {
      jq.run_jobs(false);
    }
    int k = 0;


  } else {

    std::cout << "Processing File\n";

    pathCam::Image *image = new pathCam::Image(6464,4852,2190);

    image->set_disk_file(inFile);
    image->load_raw_from_disk();


    if (!image->in_memory()) {
      std::cout << "Issue loading image.\n";
      return -1;
    }

    image->create_reg_image(1.0, 1.0, true, cv::INTER_CUBIC, false);

    imwrite(outFile.toString(), image->get_reg_image());

    delete image;
  }

  return 0;

}
