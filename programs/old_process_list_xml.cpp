//
//  process_list_truth.cpp
//  pathCam
//
//  Created by Brian Summa on 11/11/22.
//

#include "pathCam.h"

using namespace pathCam;

using Poco::Util::Application;
using Poco::Util::Option;
using Poco::Util::OptionSet;
using Poco::Util::HelpFormatter;
using Poco::Util::AbstractConfiguration;
using Poco::Util::OptionCallback;
using Poco::AutoPtr;


class BatchCamApp : public Application {
public:
  BatchCamApp() : _helpRequested(false) {
  }

protected:
  StreamCam *bcam;

  void initialize(Application &self) {
    Application::initialize(self);
    if (!_helpRequested) { bcam = new StreamCam(configPtr()); }
  }

  void uninitialize() {
    if (!_helpRequested && bcam) { delete bcam; }
    Application::uninitialize();
  }

  void reinitialize(Application &self) {
    Application::reinitialize(self);
  }

  void defineOptions(OptionSet &options) {
    Application::defineOptions(options);

    options.addOption(
      Option("help", "h", "display help information on command line arguments")
      .required(false)
      .repeatable(false)
      .callback(OptionCallback<BatchCamApp>(this, &BatchCamApp::handleHelp)));

    options.addOption(
      Option("config-file", "c", "load configuration data from a file")
      .required(true)
      .repeatable(true)
      .argument("file")
      .callback(OptionCallback<BatchCamApp>(this, &BatchCamApp::handleConfig)));
  }

  void handleHelp(const std::string &name, const std::string &value) {
    _helpRequested = true;
    displayHelp();
    stopOptionsProcessing();
  }


  void handleConfig(const std::string &name, const std::string &value) {
    loadConfiguration(value);
  }

  void displayHelp() {
    HelpFormatter helpFormatter(options());
    helpFormatter.setCommand(commandName());
    helpFormatter.setUsage("OPTIONS");
    helpFormatter.setHeader("Pathcam batch image processing application.");
    helpFormatter.format(std::cout);
  }


  const std::string &random_element(const std::vector<std::string> &v) {
    if (v.empty())
      throw std::runtime_error("random_element: vector is empty");

    thread_local static std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<std::size_t> dist(0, v.size() - 1);

    return v[dist(rng)];
  }

  int main(const ArgVec &args) {
    std::vector<std::string> v = {
      "/home/cm/Documents/data/blur_test/config/input_2_4x.txt",
      "/home/cm/Documents/data/blur_test/config/input_2x.txt",
      "/home/cm/Documents/data/blur_test/config/input_2_4_10.txt",
      "/home/cm/Documents/data/blur_test/config/input_2_4x_short.txt",
      "/home/cm/Documents/data/total_alignment_failure/config/input.txt",
      "/home/cm/Documents/data/low_feat_10x/config/input.txt",
      "/home/cm/Documents/data/low_feat_10x/config/input_2x.txt"
    };
    // std::vector<std::string> v = {
    //   "/home/cm/Documents/data/blur_test/config/input_2_4_10.txt"
    // };

    if (!_helpRequested) {
      for (int i = 0; i < 20; ++i) {
        assert(bcam);
        auto input = random_element(v);
        std::cout<<"beginning with input file "<<input<<std::endl<<std::endl;
        bcam->set_input_file(input);
        bcam->set_slide_label();
        bcam->run();
      }
      int k = 0;
    }
    return Application::EXIT_OK;
  }

private:
  bool _helpRequested;
};


POCO_APP_MAIN(BatchCamApp)
