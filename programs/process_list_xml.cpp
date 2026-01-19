//
//  new_process_list_xml.cpp
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


class FastCamApp: public Application
{
public:
  FastCamApp(): _helpRequested(false){
  }

protected:
  StreamCam *bcam;

  void initialize(Application& self){
    Application::initialize(self);
    // if(!_helpRequested){ bcam = new StreamCam(configPtr()); }
  }
  
  void uninitialize(){
    if(!_helpRequested){ delete bcam; }
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
        .callback(OptionCallback<FastCamApp>(this, &FastCamApp::handleHelp)));
        
    options.addOption(
      Option("config-file", "c", "load configuration data from a file")
        .required(true)
        .repeatable(true)
        .argument("file")
        .callback(OptionCallback<FastCamApp>(this, &FastCamApp::handleConfig)));

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
    helpFormatter.setHeader("Pathcam Fast batch image processing application.");
    helpFormatter.format(std::cout);
  }
  
  
  int main(const ArgVec& args){
    if (!_helpRequested)
    {
      int i = 0;
      while (true) {
        bcam = new StreamCam(configPtr());
        bcam->run();
        delete bcam;
        std::cout<<i++<<std::endl;
      }
      // if(!bcam->run()){ return Application::EXIT_SOFTWARE; }
    }
    return Application::EXIT_OK;
  }
  
private:
  bool _helpRequested;
};


POCO_APP_MAIN(FastCamApp)
