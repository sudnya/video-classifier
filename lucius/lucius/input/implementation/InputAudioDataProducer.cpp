/*  \file   InputAudioDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputAudioDataProducer class.
*/

// Lucius Includes
#include <lucius/input/interface/InputAudioDataProducer.h>

#include <lucius/database/interface/SampleDatabase.h>
#include <lucius/database/interface/Sample.h>

#include <lucius/network/interface/Bundle.h>

#include <lucius/audio/interface/AudioVector.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>

#include <lucius/model/interface/Model.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <random>
#include <fstream>
#include <algorithm>
#include <set>

namespace lucius
{

namespace input
{

typedef audio::Audio         Audio;
typedef audio::AudioVector   AudioVector;
typedef network::Bundle      Bundle;
typedef matrix::MatrixVector MatrixVector;

typedef std::vector<size_t> IndexVector;

static IndexVector getInputLengths(const AudioVector& batch, size_t frameSize)
{
    IndexVector result;

    for(auto& audio : batch)
    {
        result.push_back((audio.getUnpaddedLength() + frameSize - 1) / frameSize);
    }

    return result;
}

class InputAudioDataProducerImplementation
{
public:
    InputAudioDataProducerImplementation(InputAudioDataProducer* producer,
        const std::string& audioDatabaseFilename)
    : _producer(producer), _databaseFilename(audioDatabaseFilename),
      _noiseRateLower(0.0), _noiseRateUpper(0.0),
      _speechScaleLower(0.0), _speechScaleUpper(0.0),
      _sampleCache(producer)
    {

    }

public:
    void initialize()
    {
        if(_sampleCache.initialized())
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

        _noiseRateLower = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::NoiseRateLower", 0.0);
        _noiseRateUpper = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::NoiseRateUpper", 0.2);

        _speechScaleLower = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::SpeechScaleLower", 0.5);
        _speechScaleUpper = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::SpeechScaleUpper", 5.0);

        _parseAudioDatabase();

        _sampleCache.initialize();
    }

public:
    Bundle pop()
    {
        auto batch = _getBatch();

        auto audioDimension = _producer->getInputSize();

        auto input = batch.getFeatureMatrixForFrameSize(audioDimension[0]);

        auto reference = batch.getReferenceLabels(_producer->getOutputLabels(), audioDimension[0]);

        if(_producer->getStandardizeInput())
        {
            _producer->standardize(input);
        }

        util::log("InputAudioDataProducer") << "Loaded batch of '" << batch.size()
            <<  "' audio samples (" << input.size()[2] << " timesteps), "
            << _sampleCache.getRemainingSamples() << " remaining in this epoch.\n";

        return Bundle(
            std::make_pair("inputActivations", MatrixVector({input})),
            std::make_pair("inputTimesteps", getInputLengths(batch, audioDimension[0])),
            std::make_pair("referenceLabels", reference)
        );
    }

    bool empty() const
    {
        return _sampleCache.empty();
    }

public:
    size_t getUniqueSampleCount()
    {
        return _sampleCache.getUniqueSampleCount();
    }

public:
    void reset()
    {
        _sampleCache.reset();
    }

private:
    AudioVector _getBatch()
    {
        AudioVector batch;

        auto batchSize = _producer->getBatchSize();

        for(size_t i = 0; i < batchSize; ++i)
        {
            Audio audio = _sampleCache.getNextSample();

            _downsample(audio);

            _scaleRandomly(audio, _speechScaleLower, _speechScaleUpper);

            auto audioGraphemes = _toGraphemes(audio.label());
            _applyGraphemesToRange(audio, 0, audioGraphemes.size(), audioGraphemes);
            audio.setDefaultLabel("");

            batch.push_back(audio);

            if(_saveSamples)
            {
                std::stringstream filename;

                filename << "audio" << _sampleCache.getRemainingSamples() << ".wav";

                std::ofstream file(filename.str());

                batch.back().save(file, ".wav");
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
            result.setDefaultLabel("");

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

        assert(_usesGraphemes());

        auto noiseGraphemes = _toGraphemes(noise.label());
        auto audioGraphemes = _toGraphemes(audio.label());

        _applyGraphemesToRange(result, 0, noiseGraphemes.size(), noiseGraphemes);
        _applyGraphemesToRange(result, noiseGraphemes.size(),
            noiseGraphemes.size() + audioGraphemes.size(), audioGraphemes);
        _applyGraphemesToRange(result, noiseGraphemes.size() + audioGraphemes.size(),
            (result.size() - frameSize)/frameSize, noiseGraphemes);
        result.setDefaultLabel("");

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

        audio = audio.powerNormalize();

        audio.setUnpaddedLength(audio.size());
    }

    void _scaleRandomly(Audio& audio, double lower, double upper)
    {
        if(lower == 1.0 && upper == 1.0)
        {
            return;
        }

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

                if(sample.label() == "noise" || sample.label().find("|") != std::string::npos)
                {
                    if(useNoise)
                    {
                        _sampleCache.addNoise(
                            Audio(sample.path(), util::strip(sample.label(), "|")));
                    }
                    else
                    {
                        _sampleCache.addRepeatedAudio(
                            Audio(sample.path(), util::strip(sample.label(), "|")));
                    }
                }
                else
                {
                    _sampleCache.addAudio(Audio(sample.path(), sample.label()));
                }
            }
        }

        for(size_t output = 0; output != _producer->getModel()->getOutputCount(); ++output)
        {
            _graphemes.insert(_producer->getModel()->getOutputLabel(output));
        }
    }

private:
    InputAudioDataProducer* _producer;

private:
    std::string _databaseFilename;

private:
    typedef std::set<std::string> StringSet;

private:
    StringSet _graphemes;

private:
    std::default_random_engine _generator;

private:
    bool _saveSamples;

private:
    double _noiseRateLower;
    double _noiseRateUpper;

private:
    double _speechScaleLower;
    double _speechScaleUpper;

private:
    class Sample
    {
    public:
        enum Type
        {
            AudioType,
            NoiseType,
            RepeatedType
        };

    public:
        Sample(Type type, size_t audioVectorOffset, size_t audioClipOffset,
            size_t audioClipDuration)
        : type(type), audioVectorOffset(audioVectorOffset), audioClipOffset(audioClipOffset),
          audioClipDuration(audioClipDuration)
        {

        }

    public:
        Type   type;
        size_t audioVectorOffset;
        size_t audioClipOffset;
        size_t audioClipDuration;
    };

    typedef std::vector<Sample> SampleVector;
    typedef std::set<size_t> CacheSet;

private:
    class SampleCache
    {
    public:
        SampleCache(InputAudioDataProducer* producer)
        : _producer(producer), _initialized(false)
        {
            _randomWindow = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::RandomShuffleWindow", 512.);

            _totalTimestepsPerRepeat = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::TotalTimestepsPerRepeat", 512);
            _totalTimestepsPerNoise = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::TotalTimestepsPerNoise", 512);
            _totalTimestepsPerUtterance = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::TotalTimestepsPerUtterance", 512);
            _audioTimesteps             = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::AudioTimesteps", 64);

            // Determine how many audio samples to cache
            _samplesToCache = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::CacheSize", 128);

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
        }

        Audio getNextSample()
        {
            assert(_nextSample < getUniqueSampleCount());

            auto sample = _samples[_nextSample++];

            size_t begin = sample.audioClipOffset * _getFrameSize();

            Audio result;

            if(sample.type == Sample::AudioType)
            {
                result = _audio[sample.audioVectorOffset];
            }
            else if(sample.type == Sample::NoiseType)
            {
                size_t end = (sample.audioClipOffset + _totalTimestepsPerNoise) *
                    _getFrameSize();
                result = _noise[sample.audioVectorOffset].slice(begin, end);
            }
            else
            {
                assert(sample.type == Sample::RepeatedType);
                size_t end = (sample.audioClipOffset + _totalTimestepsPerRepeat) *
                    _getFrameSize();
                result = _repeatedAudio[sample.audioVectorOffset].slice(begin, end);
            }

            _updateCache();

            return result;
        }

    public:
        void addAudio(const Audio& audio)
        {
            size_t index = _audio.size();

            _audio.push_back(audio);

            _audio.back().cacheHeader();

            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

            size_t frameCount = _audio.back().duration() * frequency / _getFrameSize();

            if(frameCount <= _totalTimestepsPerUtterance)
            {
                _samples.push_back(Sample(Sample::AudioType, index, 0, frameCount));
            }
        }

        void addNoise(const Audio& noise)
        {
            size_t index = _noise.size();

            _noise.push_back(noise);

            _noise.back().cacheHeader();

            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

            size_t frameCount = _noise.back().duration() * frequency / _getFrameSize();

            size_t totalSamples = frameCount / _totalTimestepsPerNoise;

            for(size_t sample = 0; sample < totalSamples; ++sample)
            {
                _samples.push_back(Sample(Sample::NoiseType, index,
                    sample * _totalTimestepsPerNoise, _totalTimestepsPerNoise));
            }
        }

        void addRepeatedAudio(const Audio& audio)
        {
            size_t index = _repeatedAudio.size();

            _repeatedAudio.push_back(audio);

            _repeatedAudio.back().cacheHeader();

            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

            size_t frameCount = _repeatedAudio.back().duration() * frequency / _getFrameSize();

            size_t totalSamples = frameCount / _totalTimestepsPerRepeat;

            for(size_t sample = 0; sample < totalSamples; ++sample)
            {
                _samples.push_back(Sample(Sample::RepeatedType, index,
                    sample * _totalTimestepsPerRepeat, _totalTimestepsPerRepeat));
            }
        }

        size_t getUniqueSampleCount() const
        {
            size_t samples = std::min(_producer->getMaximumSamplesToRun(), _samples.size());

            size_t remainder = samples % _producer->getBatchSize();

            return samples - remainder;
        }

        size_t getRemainingSamples() const
        {
            return getUniqueSampleCount() - _nextSample;
        }

        void reset()
        {
            size_t remainingSamples = getUniqueSampleCount();
                  _nextSample       = 0;

            std::shuffle(_samples.begin(), _samples.begin() + remainingSamples, _generator);

            for(size_t begin = 0; begin < remainingSamples; begin += _randomWindow)
            {
                size_t end = std::min(begin + _randomWindow, remainingSamples);

                std::sort(_samples.begin() + begin, _samples.begin() + end,
                    [](const Sample& left, const Sample& right)
                    {
                        return left.audioClipDuration < right.audioClipDuration;
                    });
            }

            _updateCache();
        }

        void initialize()
        {
            reset();

            _sortSamplesByLength();

            _updateCache();

            _initialized = true;
        }

        bool initialized() const
        {
            return _initialized;
        }

        bool empty() const
        {
            return _nextSample >= getUniqueSampleCount();
        }

    private:
        void _updateCache()
        {
            _updateCache(_audio,         _cachedAudio,         Sample::AudioType);
            _updateCache(_noise,         _cachedNoise,         Sample::NoiseType);
            _updateCache(_repeatedAudio, _cachedRepeatedAudio, Sample::RepeatedType);
        }

        void _updateCache(AudioVector& audio, CacheSet& cacheSet, Sample::Type type)
        {
            size_t begin = _nextSample;
            size_t end = std::min(getUniqueSampleCount(),
                begin + std::max((size_t)1, _samplesToCache));

            CacheSet newCacheSet;

            for(size_t i = begin; i < end; ++i)
            {
                if(_samples[i].type != type)
                {
                    continue;
                }

                newCacheSet.insert(_samples[i].audioVectorOffset);
            }

            for(auto& element : cacheSet)
            {
                if(newCacheSet.count(element) == 0)
                {
                    audio[element].invalidateCache();
                }
            }

            for(auto& element : newCacheSet)
            {
                if(cacheSet.count(element) == 0)
                {
                    audio[element].cache();
                }
            }

            cacheSet = std::move(newCacheSet);
        }

    private:
        size_t _getFrameSize() const
        {
            return _producer->getInputSize()[0];
        }

    private:
        void _sortSamplesByLength()
        {
            std::sort(_samples.begin(), _samples.end(), [](const Sample& left, const Sample& right)
                {
                    return left.audioClipDuration < right.audioClipDuration;
                });
        }

    private:
        InputAudioDataProducer* _producer;

    private:
        AudioVector _audio;
        AudioVector _noise;
        AudioVector _repeatedAudio;

    private:
        SampleVector _samples;

    private:
        CacheSet _cachedAudio;
        CacheSet _cachedNoise;
        CacheSet _cachedRepeatedAudio;

    private:
        std::default_random_engine _generator;

    private:
        size_t _randomWindow;
        size_t _totalTimestepsPerUtterance;
        size_t _totalTimestepsPerRepeat;
        size_t _totalTimestepsPerNoise;
        size_t _audioTimesteps;
        size_t _samplesToCache;

    private:
        size_t _nextSample;

    private:
        bool _initialized;
    };

private:
    SampleCache _sampleCache;

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

InputAudioDataProducer::Bundle InputAudioDataProducer::pop()
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



