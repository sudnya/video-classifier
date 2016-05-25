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

#include <lucius/model/interface/Model.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <random>
#include <fstream>
#include <algorithm>

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
    : _producer(producer), _databaseFilename(audioDatabaseFilename), _initialized(false),
      _remainingSamples(0), _nextSample(0), _nextNoiseSample(0), _totalTimestepsPerUtterance(0),
      _audioTimesteps(0), _noiseRateLower(0.0), _noiseRateUpper(0.0),
      _speechScaleLower(0.0), _speechScaleUpper(0.0)
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

        _saveSamples = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::SaveSamples", false);

        _totalTimestepsPerUtterance = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::TotalTimestepsPerUtterance", 512);
        _audioTimesteps             = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::AudioTimesteps", 64);

        _noiseRateLower = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::NoiseRateLower", 0.0);
        _noiseRateUpper = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::NoiseRateUpper", 0.2);

        _speechScaleLower = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::SpeechScaleLower", 0.5);
        _speechScaleUpper = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::SpeechScaleUpper", 5.0);

        _parseAudioDatabase();

        // Determine how many audio samples to cache
        size_t samplesToCache = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::CacheSize", 0);

        size_t audioSamplesToCache = std::min(samplesToCache, _audio.size());

        for(size_t i = 0; i < audioSamplesToCache; ++i)
        {
            _audio[i].cache();
        }

        size_t noiseSamplesToCache = std::min(samplesToCache, _noise.size());

        for(size_t i = 0; i < noiseSamplesToCache; ++i)
        {
            _noise[i].cache();
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

        auto reference = batch.getReference(_producer->getOutputLabels(), audioDimension[0]);

        if(_producer->getStandardizeInput())
        {
            _producer->standardize(input);
        }

        util::log("InputAudioDataProducer") << "Loaded batch of '" << batch.size()
            <<  "' audio samples (" << input.size()[2] << " timesteps), "
            << _remainingSamples << " remaining in this epoch.\n";

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
        size_t samples = std::min(_producer->getMaximumSamplesToRun(), _audio.size());

        size_t remainder = samples % _producer->getBatchSize();

        return samples - remainder;
    }

public:
    void reset()
    {
        _remainingSamples = getUniqueSampleCount();
        _nextSample       = 0;
        _nextNoiseSample  = 0;

        std::shuffle(_audio.begin(), _audio.begin() + _remainingSamples, _generator);
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
            bool isAudioCached = _audio[_nextSample].isCached();
            bool isNoiseCached = false;

            _audio[_nextSample].cache();

            auto audio = _audio[_nextSample];

            if(!_noise.empty())
            {
                isNoiseCached = _noise[_nextNoiseSample].isCached();

                _noise[_nextNoiseSample].cache();

                auto noise = _noise[_nextNoiseSample];

                _downsampleToMatchFrequencies(audio, noise);

                _scaleRandomly(audio, _speechScaleLower, _speechScaleUpper);
                _scaleRandomly(noise, _noiseRateLower,   _noiseRateUpper);

                auto selectedTimesteps = ((_generator() % _audioTimesteps) + _audioTimesteps) / 2;

                audio = _sample(audio, selectedTimesteps          );
                noise = _sample(noise, _totalTimestepsPerUtterance);

                batch.push_back(_merge(audio, noise));
            }
            else
            {
                _downsample(audio);

                _scaleRandomly(audio, _speechScaleLower, _speechScaleUpper);

                auto audioGraphemes = _toGraphemes(audio.label());
                _applyGraphemesToRange(audio, 0, audioGraphemes.size(), audioGraphemes);

                _setSpecialGraphemes(audio, audioGraphemes.size());

                batch.push_back(audio);
            }

            if(_saveSamples)
            {
                std::stringstream filename;

                filename << "audio" << _nextSample << "noise" << _nextNoiseSample << ".wav";

                std::ofstream file(filename.str());

                batch.back().save(file, ".wav");

                std::stringstream audiofilename;

                audiofilename << "audio" << _nextSample << ".wav";

                std::ofstream audiofile(audiofilename.str());

                _audio[_nextSample].save(audiofile, ".wav");
            }

            if(!isAudioCached)
            {
                _audio[_nextSample].invalidateCache();
            }

            if(!isNoiseCached && !_noise.empty())
            {
                _noise[_nextNoiseSample].invalidateCache();
            }

            --_remainingSamples;
            ++_nextSample;

            if(!_noise.empty())
            {
                _nextNoiseSample = (_nextNoiseSample + 1) % _noise.size();
            }
        }

        return batch;
    }

    Audio _sample(const Audio& audio, size_t timesteps)
    {
        size_t samples = timesteps * _getFrameSize();

        if(samples >= audio.size())
        {
            auto copy = audio;

            _zeroPad(copy, samples);

            return copy;
        }

        size_t remainder = audio.size() - samples;

        size_t position = _generator() % remainder;

        return audio.slice(position, position + samples);
    }

    void _zeroPad(Audio& audio, size_t samples) const
    {
        size_t position = audio.size();

        audio.resize(samples);

        audio.setUnpaddedLength(position);

        for(; position < audio.size(); ++position)
        {
            audio.setSample(position, 0.0);
        }
    }

    Audio _merge(const Audio& audio, const Audio& noise)
    {
        if(audio.size() * 2 >= noise.size())
        {
            auto result = audio;

            auto audioGraphemes = _toGraphemes(audio.label());
            _applyGraphemesToRange(result, 0, audioGraphemes.size(), audioGraphemes);

            _setSpecialGraphemes(result, audioGraphemes.size());

            return result;
        }

        assert(2 * audio.size() < noise.size());

        size_t remainder = noise.size() - 2 * audio.size();

        auto result = noise;

        result.clearLabels();

        size_t position = audio.size() + (_generator() % remainder);

        for(size_t existingPosition = 0; existingPosition != audio.size(); ++existingPosition)
        {
            result.setSample(position, audio.getSample(existingPosition) +
                result.getSample(position));

            ++position;
        }

        size_t frameSize = _getFrameSize();

        if(_usesGraphemes())
        {
            auto noiseGraphemes = _toGraphemes(noise.label());
            auto audioGraphemes = _toGraphemes(audio.label());

            _applyGraphemesToRange(result, 0, noiseGraphemes.size(), noiseGraphemes);
            _applyGraphemesToRange(result, noiseGraphemes.size(),
                noiseGraphemes.size() + audioGraphemes.size(), audioGraphemes);
            _applyGraphemesToRange(result, noiseGraphemes.size() + audioGraphemes.size(),
                (result.size() - frameSize)/frameSize, noiseGraphemes);

            _setSpecialGraphemes(result, noiseGraphemes.size() + audioGraphemes.size());
        }
        else
        {
            result.addLabel(0, position * frameSize, noise.label());
            result.addLabel(position * frameSize,
                frameSize * (position + audio.size()), audio.label());
            result.addLabel(frameSize * (position + audio.size()),
                noise.size() * frameSize, noise.label());
        }

        return result;
    }

private:
    typedef std::vector<std::string> StringVector;

private:
    void _applyGraphemesToRange(Audio& result, size_t beginTimestep, size_t endTimestep,
        const StringVector& graphemes)
    {
        size_t limit = std::min(beginTimestep + graphemes.size(), endTimestep);

        size_t frameSize = _getFrameSize();

        for(size_t i = beginTimestep, grapheme = 0; i < limit; ++i, ++grapheme)
        {
            result.addLabel(frameSize * i, frameSize*(i+1), graphemes[grapheme]);
        }
    }

    StringVector _toGraphemes(const std::string& label)
    {
        if(_graphemes.empty())
        {
            throw std::runtime_error("Could not match remaining label '" + label +
                "' against a grapheme because there aren't any graphemes.");
        }

        auto remainingLabel = label;

        StringVector graphemes;

        while(!remainingLabel.empty())
        {
            auto insertPosition = _graphemes.lower_bound(remainingLabel);

            // exact match
            if(insertPosition != _graphemes.end() && *insertPosition == remainingLabel)
            {
                graphemes.push_back(remainingLabel);
                break;
            }

            // ordered before first grapheme
            if(insertPosition == _graphemes.begin())
            {
                throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                    "' against a grapheme.");
            }

            --insertPosition;

            auto grapheme = remainingLabel.substr(0, insertPosition->size());

            if(grapheme != *insertPosition)
            {
                throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                    "' against best grapheme '" + *insertPosition + "'.");
            }

            graphemes.push_back(grapheme);

            remainingLabel = remainingLabel.substr(grapheme.size());
        }

        return graphemes;
    }

    void _setSpecialGraphemes(Audio& result, size_t sequenceLength)
    {
        result.setDefaultLabel(*_graphemes.begin());

        size_t frameSize = _getFrameSize();

        size_t endAudioFrame = result.getUnpaddedLength() / frameSize;

        size_t beginAudioEnd = endAudioFrame * frameSize;
        size_t endAudioEnd   = std::min(result.size(), (endAudioFrame + 1) * frameSize);

        result.addLabel(beginAudioEnd, endAudioEnd, _delimiterGrapheme);

        size_t endLabelFrame = sequenceLength + 1;

        size_t beginLabelEnd = endLabelFrame * frameSize;
        size_t endLabelEnd   = std::min(result.size(), (endLabelFrame + 1) * frameSize);

        result.addLabel(beginLabelEnd, endLabelEnd, _delimiterGrapheme);
    }

    bool _usesGraphemes() const
    {
        return !_graphemes.empty();
    }

    void _downsampleToMatchFrequencies(Audio& first, Audio& second)
    {
        _downsample(first);
        _downsample(second);
    }

    void _downsample(Audio& audio)
    {
        size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

        if(audio.frequency() != frequency)
        {
            audio = audio.sample(frequency);
        }

        audio.setUnpaddedLength(audio.size());
    }

    void _scaleRandomly(Audio& audio, double lower, double upper)
    {
        std::uniform_real_distribution<double> distribution(lower, upper);

        double scale = distribution(_generator);

        for(size_t sample = 0; sample != audio.size(); ++sample)
        {
            audio.setSample(sample, scale * audio.getSample(sample));
        }
    }

private:
    size_t _getFrameSize() const
    {
        return _producer->getInputSize()[0];
    }

private:
    void _parseAudioDatabase()
    {
        util::log("InputAudioDataProducer") << " scanning audio database '"
            << _databaseFilename << "'\n";

        database::SampleDatabase sampleDatabase(_databaseFilename);

        sampleDatabase.load();

        bool useNoise = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::UseNoise", false);

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

                if(sample.label() == "noise" || sample.label().find("background|") == 0)
                {
                    if(useNoise)
                    {
                        _noise.push_back(Audio(sample.path(), util::strip(sample.label(), "|")));
                    }
                }
                else
                {
                    _audio.push_back(Audio(sample.path(), sample.label()));
                }
            }
        }

        for(size_t output = 0; output != _producer->getModel()->getOutputCount(); ++output)
        {
            _graphemes.insert(_producer->getModel()->getOutputLabel(output));
        }

        _delimiterGrapheme = _producer->getModel()->getAttribute<std::string>("DelimiterGrapheme");
    }

private:
    InputAudioDataProducer* _producer;

private:
    std::string _databaseFilename;

private:
    AudioVector _audio;
    AudioVector _noise;

private:
    typedef std::set<std::string> StringSet;

private:
    StringSet _graphemes;
    std::string _delimiterGrapheme;

private:
    std::default_random_engine _generator;

private:
    bool _initialized;
    bool _saveSamples;

private:
    size_t _remainingSamples;
    size_t _nextSample;
    size_t _nextNoiseSample;

private:
    size_t _totalTimestepsPerUtterance;
    size_t _audioTimesteps;

private:
    double _noiseRateLower;
    double _noiseRateUpper;

private:
    double _speechScaleLower;
    double _speechScaleUpper;

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



