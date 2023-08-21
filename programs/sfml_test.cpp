
////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/Audio.hpp>
#include <pathCam.h>

#include <iostream>
#include <sstream>

using Poco::Path;

class BeginSound: public Poco::Runnable {
public:
  sf::Sound *sound;
  sf::SoundBuffer *buffer;
  
  BeginSound():sound(0), buffer(0){
    
    std::stringstream ss;
    ss <<  PROJECT_SOURCE_DIR << "/resources/start.wav";

    Poco::Path start_wav_path = Poco::Path(ss.str());
    
    buffer = new sf::SoundBuffer();
      
    if (buffer->loadFromFile(start_wav_path.toString())){
      sound = new sf::Sound(*buffer);
    }
  }

  ~BeginSound(){
    if(sound){delete sound;}
    if(buffer){delete buffer;}
  }
  
  virtual void run(){
    std::cout << "Hello, world!" << std::endl;
    if(sound){
      sound->play();
      while (sound->getStatus() == sf::Sound::Playing){
        // Leave some CPU time for other processes
        Poco::Thread::sleep(100);
      }
    }
  }
};

class EndSound: public Poco::Runnable {
public:
  sf::Sound *sound;
  sf::SoundBuffer *buffer;
  
  EndSound():sound(0), buffer(0){
    
    std::stringstream ss;
    ss <<  PROJECT_SOURCE_DIR << "/resources/stop.wav";

    Poco::Path start_wav_path = Poco::Path(ss.str());
    
    buffer = new sf::SoundBuffer();
      
    if (buffer->loadFromFile(start_wav_path.toString())){
      sound = new sf::Sound(*buffer);
    }
  }

  ~EndSound(){
    if(sound){delete sound;}
    if(buffer){delete buffer;}
  }
  
  virtual void run(){
    std::cout << "Hello, world!" << std::endl;
    if(sound){
      sound->play();
      while (sound->getStatus() == sf::Sound::Playing){
        // Leave some CPU time for other processes
        Poco::Thread::sleep(100);
      }
    }
  }
};


int main(int argc, char** argv)
{
  BeginSound start;
  Poco::Thread thread_start;
  thread_start.start(start);
  
  EndSound stop;
  Poco::Thread thread_stop;
  thread_stop.start(stop);

  thread_start.join();
  thread_stop.join();
  return 0;
}




//////////////////////////////////////////////////////////////
///// Play a sound
/////
//////////////////////////////////////////////////////////////
//void playSound()
//{
//
//  std::stringstream ss;
//  ss <<  PROJECT_SOURCE_DIR << "/resources/stop.wav";
//
//  Poco::Path start_wav_path = Poco::Path(ss.str());
//
//  // Load a sound buffer from a wav file
//    sf::SoundBuffer buffer;
//    if (!buffer.loadFromFile(start_wav_path.toString()))
//        return;
//
//    // Create a sound instance and play it
//    sf::Sound sound(buffer);
//    sound.play();
//
//    // Loop while the sound is playing
//
//    std::cout << std::endl << std::endl;
//}
//
//
//
//int main()
//{
//    // Play a sound
//    playSound();
//
//    // Wait until the user presses 'enter' key
//    std::cout << "Press enter to exit..." << std::endl;
//    std::cin.ignore(10000, '\n');
//}
