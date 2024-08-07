//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"
#include "Poco/DirectoryIterator.h"

using Poco::DirectoryIterator;

unsigned long extractSortableValue(const std::string &str) {
  if (str.size() <= 4) {
    // Handle cases where the string is too short
    return 0;
  }
  std::string substr = str.substr(0, str.size() - 4);
  return std::stoul(substr);
}

int extractSortableValue2(const std::string &str) {
  size_t firstDash = str.find('-');
  if (firstDash == std::string::npos) {
    throw std::invalid_argument("doesn't have a dash");
  }

  size_t secondDash = str.find('-', firstDash + 1);
  if (secondDash == std::string::npos) {
    throw std::invalid_argument("doesn't have a second dash");
  }

  size_t period = str.find('.', secondDash + 1);
  if (period == std::string::npos) {
    throw std::invalid_argument("doesn't have a period");
  }

  return std::stoi(str.substr(secondDash + 1, period - secondDash - 1));
}

bool customComparator2(const pathCam::Image *lhs, const pathCam::Image *rhs) {
  unsigned long lhsVal = extractSortableValue2(lhs->image_file.getFileName());
  unsigned long rhsVal = extractSortableValue2(rhs->image_file.getFileName());
  return lhsVal < rhsVal;
}

bool customComparator(const pathCam::Image *lhs, const pathCam::Image *rhs) {
  unsigned long lhsVal = extractSortableValue(lhs->image_file.getFileName());
  unsigned long rhsVal = extractSortableValue(rhs->image_file.getFileName());
  return lhsVal < rhsVal;
}

int main(int argc, char *argv[]) {
  Mat flat_field;
  flat_field = cv::imread("/Users/coopermaira/Desktop/pathcam_data/2x_wb.tif");
  flat_field.convertTo(flat_field, CV_32F);
  flat_field *= 1 / 170.0;
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
    std::vector<std::string> *names = new std::vector<std::string>;

    blur->resize(3000, 0);
    names->resize(3000);

    std::cout << "Processing Directories\n";
    auto jq = pathCam::JobQueue(10, 10);

    Poco::DirectoryIterator it(inFile);
    Poco::DirectoryIterator end;
    std::vector<pathCam::Image *> images;
    while (it != end) {

      Poco::Path p(it.path());


      if (p.getExtension() == "Raw") {
        //std::cout << "read:" << p.toString() << "\n";

        auto *image = new pathCam::Image(6464, 4852, 2190);
        image->set_disk_file(p);
        images.push_back(image);

      }
      ++it;
    }
    std::sort(images.begin(), images.end(), customComparator);


    std::string outputfilepath = "/Users/coopermaira/Desktop/pathcam_data/raw2to40run/input.txt";
    std::ofstream outputFile(outputfilepath);

    for (int i = 0; i < images.size(); i++) {
//      Poco::File currentFile(images[i]->image_file);
//      if(currentFile.isFile()){
//        std::string newName = std::to_string(i)+".Raw";
//        Poco::Path newfilepath = Poco::Path(argv[1]);
//        newfilepath.setFileName(newName);
//        currentFile.renameTo(newfilepath.toString());
//        int k = 0;
//      }
      outputFile << images[i]->image_file.toString()<<std::endl;
//      auto *dr = new pathCam::DebayerRunnable(images[i], flat_field, outFile, blur, names, i);
//      jq.add_runnable(dr, i);
    }
    outputFile.close();
    while (!jq.is_empty()) {
      jq.run_jobs(false);
    }
    int k = 0;


  } else {

    std::cout << "Processing File\n";

    pathCam::Image *image = new pathCam::Image(6464, 4852, 2190);

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


