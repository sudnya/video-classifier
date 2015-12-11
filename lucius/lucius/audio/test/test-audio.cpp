/*! \file   test-audio.cpp
    \author Gregory Diamos <solusstultus@gmail.com>
    \date   August 18, 2015
    \brief  A unit test for lucius audio support.
*/

// Lucius Includes
#include <lucius/audio/interface/Audio.h>

// Standard Library Includes
#include <cmath>
#include <sstream>
#include <iostream>

static lucius::audio::Audio generateSimpleTone()
{
    size_t frequency = 44100;
    size_t samples   = 10 * frequency;

    lucius::audio::Audio sample(samples, 2, frequency);

    for(size_t i = 0; i < samples; ++i)
    {
        sample.setSample(i, std::sin(i * 2.0 * 3.14 * 440 / frequency) * 10000.0);
    }

    return sample;
}

static bool testGenericLossless(const std::string& format)
{
    auto simpleTone = generateSimpleTone();

    std::stringstream stream;

    simpleTone.save(stream, format);

    lucius::audio::Audio loadedTone(stream, format);

    if(loadedTone.size() > simpleTone.size())
    {
        loadedTone.resize(simpleTone.size());
    }

    if(simpleTone != loadedTone)
    {
        std::cout << " Test Failed: Audio signal does not match for format " + format + ".\n";

        std::cout << "  Input  size is: " << simpleTone.size() << "\n";
        std::cout << "  Output size is: " << loadedTone.size() << "\n";

        std::cout << "  Input  frequency is: " << simpleTone.frequency() << "\n";
        std::cout << "  Output frequency is: " << loadedTone.frequency() << "\n";

        std::cout << "  Input  sample size is: " << simpleTone.bytesPerSample() << "\n";
        std::cout << "  Output sample size is: " << loadedTone.bytesPerSample() << "\n";

        for(size_t timestep = 0; timestep != simpleTone.size(); ++timestep)
        {
            if(simpleTone.getSample(timestep) != loadedTone.getSample(timestep))
            {
                std::cout << "  Mismatched sample " << timestep << ", original value "
                    << simpleTone.getSample(timestep) << " vs loaded value "
                    << loadedTone.getSample(timestep) << "\n";
                break;
            }
        }

        return;
    }

    std::cout << " Test " << format << " Passed\n";
}

static void runTest()
{
    bool result = true;

    result &= testGenericLossless(".flac");
    result &= testGenericLossless(".wav");

    if(result)
    {
        std::cout << "Test Passed\n";
    }
    else
    {
        std::cout << "Test Failed\n";
    }
}

int main(int argc, char** argv)
{
    runTest();

    return 0;
}




