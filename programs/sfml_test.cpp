
////////////////////////////////////////////////////////////
// Headers
////////////////////////////////////////////////////////////
#include <SFML/Audio.hpp>
#include <pathCam.h>

#include <iostream>
#include <sstream>

using Poco::Path;


////////////////////////////////////////////////////////////
/// Play a sound
///
////////////////////////////////////////////////////////////
void playSound()
{
  
  std::stringstream ss;
  ss <<  PROJECT_SOURCE_DIR << "/resources/stop.wav";

  Poco::Path start_wav_path = Poco::Path(ss.str());
  
  // Load a sound buffer from a wav file
    sf::SoundBuffer buffer;
    if (!buffer.loadFromFile(start_wav_path.toString()))
        return;

    // Create a sound instance and play it
    sf::Sound sound(buffer);
    sound.play();

    // Loop while the sound is playing
    while (sound.getStatus() == sf::Sound::Playing)
    {
        // Leave some CPU time for other processes
        sf::sleep(sf::milliseconds(100));
    }
    std::cout << std::endl << std::endl;
}



int main()
{
    // Play a sound
    playSound();

    // Wait until the user presses 'enter' key
    std::cout << "Press enter to exit..." << std::endl;
    std::cin.ignore(10000, '\n');
}
