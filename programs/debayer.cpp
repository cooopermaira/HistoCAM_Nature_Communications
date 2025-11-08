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
  auto renameFiles = true;
  auto convertImages = true;
  bool makeInput = true;

  Mat flat_field2x, flat_field4x, flat_field10x, flat_field20x, flat_field40x;
  std::ifstream stream;
  /*
  {

    stream.open("/Users/coopermaira/Desktop/pathcam_data/1at1/other_rec/cal/2x_cal.Raw", std::ios::binary);
    char *raw_buffer = new char[6464 * 4852];
    stream.read(raw_buffer, 6464 * 4852);
    stream.close();
    flat_field2x = cv::Mat(cv::Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);

  }
  cvtColor(flat_field2x, flat_field2x, COLOR_BayerBG2RGB);

  flat_field2x.convertTo(flat_field2x, CV_32F);
  flat_field2x *= 1 / 170.0;

  stream.open("/Users/coopermaira/Desktop/pathcam_data/1at1/other_rec/cal/4x_cal.Raw", std::ios::binary);

  {
    char *raw_buffer = new char[6464 * 4852];
    stream.read(raw_buffer, 6464 * 4852);
    stream.close();
    flat_field4x = cv::Mat(cv::Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);
  }
  cvtColor(flat_field4x, flat_field4x, COLOR_BayerBG2BGR);
  flat_field4x.convertTo(flat_field4x, CV_32F);
  flat_field4x *= 1 / 170.0;


  stream.open("/Users/coopermaira/Desktop/pathcam_data/1at1/other_rec/cal/10x_cal.Raw", std::ios::binary);
  {
    char *raw_buffer = new char[6464 * 4852];
    stream.read(raw_buffer, 6464 * 4852);
    stream.close();
    flat_field10x = cv::Mat(cv::Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);
  }
  cvtColor(flat_field10x, flat_field10x, COLOR_BayerBG2BGR);
  flat_field10x.convertTo(flat_field10x, CV_32F);
  flat_field10x *= 1 / 170.0;

  stream.open("/media/max/Data/2_20/20x/cal/20x_cal.Raw", std::ios::binary);
  {
    char *raw_buffer = new char[6464 * 4852];
    stream.read(raw_buffer, 6464 * 4852);
    stream.close();
    flat_field20x = cv::Mat(cv::Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);
  }
  cvtColor(flat_field20x, flat_field20x, COLOR_BayerBG2BGR);
  flat_field20x.convertTo(flat_field20x, CV_32F);
  flat_field20x *= 1 / 170.0;
*/
//  stream.open("/media/max/Data/afb/cal/40x_cal.Raw", std::ios::binary);
//  {
//    char *raw_buffer = new char[6464 * 4852];
//    stream.read(raw_buffer, 6464 * 4852);
//    stream.close();
//    flat_field40x = cv::Mat(cv::Size(6464, 4852), CV_8U, raw_buffer, Mat::AUTO_STEP);
//  }
//  cvtColor(flat_field40x, flat_field40x, COLOR_BayerBG2RGB);
//  flat_field40x.convertTo(flat_field40x, CV_32F);
//  flat_field40x *= 1 / 170.0;


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
    std::vector<int> *blur = new std::vector<int>;
    std::vector<std::string> *names = new std::vector<std::string>;

    blur->resize(5000, 0);
    names->resize(5000);

    std::cout << "Processing Directories\n";
    auto jq = pathCam::JobQueue(15, 15);

    Poco::DirectoryIterator it(inFile);
    Poco::DirectoryIterator end;
    std::vector<pathCam::Image *> images;
    while (it != end) {

      Poco::Path p(it.path());


      if (p.getExtension() == "Raw" || p.getExtension() == "raw") {
        //std::cout << "read:" << p.toString() << "\n";

        auto *image = new pathCam::Image(6464, 4852, 2190);
        image->set_disk_file(p);
        images.push_back(image);

      }
      ++it;
    }

    if (renameFiles || makeInput) {
      try {
        std::sort(images.begin(), images.end(), customComparator2);
      }catch(...){
        std::sort(images.begin(), images.end(), customComparator);
      }
    }

    auto outputfilepath = outFile;
    outputfilepath.makeParent().append("input.txt");
    std::ofstream outputFile(outputfilepath.toString());

    for (int i = 0; i < images.size(); i++) {
      if (renameFiles) {
        Poco::File currentFile(images[i]->image_file);
        if (currentFile.isFile()) {
          std::string newName = std::to_string(i) + ".Raw";
          Poco::Path newfilepath = Poco::Path(argv[1]);
          newfilepath.setFileName(newName);
          images[i]->set_disk_file(newfilepath);
          currentFile.renameTo(newfilepath.toString());
          if (makeInput) {
            outputFile << newfilepath.toString() << std::endl;
          }
          int k = 0;
        }

      }

      if (makeInput && !renameFiles) {
        outputFile << images[i]->image_file.toString() << std::endl;
      }
      if (convertImages) {
        Mat ff;

        ff = flat_field20x;
        string of = outFile.toString();
        auto *dr = new pathCam::DebayerRunnable(images[i], ff, outFile, blur, names, i);
        jq.add_runnable(dr, i);
      }
    }
    outputFile.close();

    auto start = std::chrono::high_resolution_clock::now();
    while (!jq.is_empty()) {
      jq.run_jobs(false);
    }
    jq.pool->joinAll();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << duration.count() << std::endl;
    int k = 0;
    /*auto blurfile = outFile;
    blurfile.makeParent().append("blurfile.txt");
    std::ofstream blurfilestream(outputfilepath.toString());
    for (int i = 0; i < blur->size();++i) {
      blurfilestream<<blur[i]<<" "<<names[i]<<std::endl;
    }
    blurfilestream.close();
    int k = 0;*/


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


