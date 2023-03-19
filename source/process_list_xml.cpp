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


class BatchCamApp: public Application
{
public:
  BatchCamApp(): _helpRequested(false){
  }

protected:
  BatchCam *bcam;

  void initialize(Application& self){
    Application::initialize(self);
    bcam = new BatchCam(configPtr());
  }
  
  void uninitialize(){
    delete bcam;
    Application::uninitialize();
  }
  
  void reinitialize(Application& self){
    Application::reinitialize(self);
  }
  
  void defineOptions(OptionSet& options){
    Application::defineOptions(options);

    options.addOption(
      Option("help", "h", "display help information on command line arguments")
        .required(false)
        .repeatable(false)
        .callback(OptionCallback<BatchCamApp>(this, &BatchCamApp::handleHelp)));
        
    options.addOption(
      Option("config-file", "c", "load configuration data from a file")
        .required(false)
        .repeatable(true)
        .argument("file")
        .callback(OptionCallback<BatchCamApp>(this, &BatchCamApp::handleConfig)));

  }
  
  void handleHelp(const std::string& name, const std::string& value){
    _helpRequested = true;
    displayHelp();
    stopOptionsProcessing();
  }
  
  
  void handleConfig(const std::string& name, const std::string& value){
    loadConfiguration(value);
  }
    
  void displayHelp(){
    HelpFormatter helpFormatter(options());
    helpFormatter.setCommand(commandName());
    helpFormatter.setUsage("OPTIONS");
    helpFormatter.setHeader("Pathcam batch image processing application.");
    helpFormatter.format(std::cout);
  }
  
  
  int main(const ArgVec& args){
    if (!_helpRequested)
    {
      if(!bcam->run()){ return Application::EXIT_SOFTWARE; }
    }
    return Application::EXIT_OK;
  }
  
private:
  bool _helpRequested;
};


POCO_APP_MAIN(BatchCamApp)
