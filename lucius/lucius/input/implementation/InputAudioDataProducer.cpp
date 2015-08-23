/*  \file   InputAudioDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputAudioDataProducer class.
*/


// Lucius Includes
#include <lucius/input/interface/InputAudioDataProducer.h>

#include <lucius/database/interface/SampleDatabase.h>
#include <lucius/database/interface/Sample.h>

#include <lucius/audio/interface/AudioVector.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <random>

namespace lucius
{

namespace input
{

typedef audio::Audio       Audio;
typedef audio::AudioVector AudioVector;

class InputAudioDataProducerImplementation
{
public:
    InputAudioDataProducerImplementation(InputAudioDataProducer* producer,
        const std::string& audioDatabaseFilename)
    : _producer(producer), _databaseFilename(audioDatabaseFilename),
      _remainingSamples(0), _nextSample(0), _nextNoiseSample(0)
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

        samplesToCache = std::min(samplesToCache, _audio.size());

        for(size_t i = 0; i < samplesToCache; ++i)
        {
            _audio[i].cache();
        }

        reset();

        _initialized = true;
    }

public:
    InputAudioDataProducer::InputAndReferencePair pop()
    {
        assert(_initialized);

        auto batch = _getBatch();

        auto audioDimension = _producer->getInputSize();

        auto input = batch.getFeatureMatrixForFrameSize(audioDimension[0]);

        auto reference = batch.getReference(_producer->getOutputLabels());

        if(_producer->getStandardizeInput())
        {
            _producer->standardize(input);
        }

        util::log("InputAudioDataProducer") << "Loaded batch of '" << batch.size()
            <<  "' audio samples, " << _remainingSamples << " remaining in this epoch.\n";

        return InputAudioDataProducer::InputAndReferencePair(
            std::move(input), std::move(reference));
    }

    bool empty() const
    {
        return _remainingSamples == 0;
    }

public:
    size_t getUniqueSampleCount()
    {
        return std::min(_producer->getMaximumSamplesToRun(), _audio.size());
    }

public:
    void reset()
    {
        _remainingSamples = getUniqueSampleCount();
        _nextSample       = 0;
        _nextNoiseSample  = 0;

        std::shuffle(_audio.begin(), _audio.end(), _generator);
        std::shuffle(_noise.begin(), _noise.end(), _generator);
    }

private:
    AudioVector _getBatch()
    {
        AudioVector batch;

        auto batchSize = std::min(_audio.size(), std::min(_remainingSamples,
            _producer->getBatchSize()));

        for(size_t i = 0; i < batchSize; ++i)
        {
            auto audio = _sample(_audio[_nextSample],      _audioTimesteps         );
            auto noise = _sample(_noise[_nextNoiseSample], _totalTimestepsPerSample);

            batch.push_back(_merge(audio, noise));

            --_remainingSamples;
            ++_nextSample;
            _nextNoiseSample = (_nextNoiseSample + 1) % _noise.size();
        }

        return batch;
    }

    Audio _sample(const Audio& audio, size_t timesteps)
    {
        if(timesteps <= audio.size())
        {
            auto copy = audio;

            _zeroPad(copy, timesteps);

            return copy;
        }

        size_t remainder = audio.size() - timesteps;

        size_t position = _generator() % remainder;

        return audio.slice(position, position + timesteps);
    }

    void _zeroPad(Audio& audio, size_t timesteps) const
    {
        size_t position = audio.size();

        audio.resize(timesteps);

        for(; position < audio.size(); ++position)
        {
            audio.setSample(position, 0.0);
        }
    }

    Audio _merge(const Audio& audio, const Audio& noise)
    {
        assert(audio.size() < noise.size());

        size_t remainder = noise.size() - audio.size();

        auto result = noise;

        result.clearLabels();

        size_t position = _generator() % remainder;

        for(size_t existingPosition = 0; existingPosition != audio.size(); ++existingPosition)
        {
            result.setSample(position, audio.getSample(existingPosition) +
                result.getSample(position));

            ++position;
        }

        result.addLabel(0,                       position,                noise.label());
        result.addLabel(position,                position + audio.size(), audio.label());
        result.addLabel(position + audio.size(), noise.size(),            noise.label());

        return result;
    }

private:
    void _parseAudioDatabase()
    {
        util::log("InputAudioDataProducer") << " scanning audio database '"
            << _databaseFilename << "'\n";

        database::SampleDatabase sampleDatabase(_databaseFilename);

        sampleDatabase.load();

        for(auto& sample : sampleDatabase)
        {
            if(!sample.hasLabel() && _producer->getRequiresLabeledData())
            {
                util::log("InputAudioDataProducer::Detail") << "  skipped unlabeled data '"
                    << sample.path() << "'\n";
                continue;
            }

            if(sample.isAudioSample())
            {
                if(sample.hasLabel())
                {
                    util::log("InputAudioDataProducer::Detail") << "  found labeled image '"
                        << sample.path() << "' with label '" << sample.label() << "'\n";
                }
                else
                {
                    util::log("InputAudioDataProducer::Detail") << "  found unlabeled image '"
                        << sample.path() << "'\n";
                }

                _audio.push_back(Audio(sample.path(), sample.label()));
            }
        }
    }

private:
    InputAudioDataProducer* _producer;

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
    size_t _nextSample;
    size_t _nextNoiseSample;

private:
    size_t _totalTimestepsPerSample;
    size_t _audioTimesteps;

};

InputAudioDataProducer::InputAudioDataProducer(const std::string& audioDatabaseFilename)
: _implementation(std::make_unique<InputAudioDataProducerImplementation>(
    this, audioDatabaseFilename))
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



