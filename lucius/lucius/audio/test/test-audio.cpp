/*! \file   test-audio.cpp
    \author Gregory Diamos <solusstultus@gmail.com>
    \date   August 18, 2015
    \brief  A unit test for lucius audio support.
*/

// Lucius Includes
#include <lucius/audio/interface/Audio.h>

static lucius::audio::Audio generateSimpleTone()
{
    size_t frequency = 44100;
    size_t samples   = 10 * frequency;

    lucius::audio::Audio sample(samples, 2, frequency);

    for(size_t i = 0; i < samples; ++i)
    {
        sample.setSample(i, std::sin(i) * 10000.0);
    }

    return sample;
}

static void runTest()
{
    auto simpleTone = generateSimpleTone();

    std::stringstream stream;

    simpleTone.save(stream, "flac");

    lucius::audio::Audio loadedTone(stream, "flac");

    if(simpleTone != loadedTone)
    {
        std::cout << "Test Failed: Audio signal does not match.\n";
        return;
    }

    std::cout << "Test Passed\n";
}

int main(int argc, char** argv)
{
    runTest();

    return 0;
}




