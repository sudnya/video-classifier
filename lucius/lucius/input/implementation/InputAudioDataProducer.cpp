/*    \file   InputAudioDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputAudioDataProducer class.
*/

#include <lucius/input/interface/InputAudioDataProducer.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace input
{

class InputAudioDataProducerImplementation
{
public:
    InputAudioDataProducerImplementation(const std::string& audioDatabaseFilename)
    : _databaseFilename(audioDatabaseFilename)
    {

    }

public:
    void initialize()
    {
        if(_initialized)
        {
            return;
        }

        util::log("InputAudioDataProducer") << "Initializing from audio database '"
            << _databaseFilename << "'\n";

        bool shouldSeedWithTime = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::SeedWithTime", false);

        if(shouldSeedWithTime)
        {
            _generator.seed(std::time(0));
        }
        else
        {
            _generator.seed(127);
        }

        _parseAudioDatabase();

        // Determine how many images to cache
        size_t samplesToCache = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::CacheSize", 128);

        samplesToCache = std::min(samplesToCache, _images.size());

        for(size_t i = 0; i < samplesToCache; ++i)
        {
            _audio[i].load();
        }

        reset();

        _initialized = true;
    }

private:
    void reset()
    {
        _remainingSamples = getUniqueSampleCount();

        std::shuffle(_audio.begin(), _audio.end(), _generator);
        std::shuffle(_noise.begin(), _noise.end(), _generator);
    }

private:
    std::string _databaseFilename;

private:
    AudioVector _audio;
    AudioVector _noise;

private:
    std::default_random_engine _generator;

private:
    bool _initialized;

private:
    size_t _remainingSamples;

};

InputAudioDataProducer::InputAudioDataProducer(const std::string& audioDatabaseFilename)
: _implementation(new InputAudioDataProducerImplementation(audioDatabaseFilename))
{

}

InputAudioDataProducer::~InputAudioDataProducer()
{

}

void InputAudioDataProducer::initialize()
{
    _implementation->initialize();
}

InputAudioDataProducer::InputAndReferencePair InputAudioDataProducer::pop()
{
    return _implementation->pop();
}

bool InputAudioDataProducer::empty() const
{
    return _implementation->empty();
}

void InputAudioDataProducer::reset()
{
    _implementation->reset();
}

size_t InputAudioDataProducer::getUniqueSampleCount() const
{
    return _implementation->getUniqueSampleCount();
}


}

}



