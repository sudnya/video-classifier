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

public:
    void addRawSample(const void* data, size_t size,
        const std::string& type, const std::string& label)
    {
        util::log("InputAudioDataProducer::Detail") << " adding raw sample type '"
            << type << "' and label '" << label << "'\n";
        std::stringstream stream;

        stream.write(reinterpret_cast<const char*>(data), size);

        _sampleCache.addAudio(Audio(stream, type, label));
    }

private:
    AudioVector _getBatch()
    {
        AudioVector batch;

        auto batchSize = _producer->getBatchSize();

        util::log("InputAudioDataProducer::Detail") << "Getting next mini batch with size '"
            << batchSize << "'\n";

        for(size_t i = 0; i < batchSize; ++i)
        {
            Audio audio = _sampleCache.getNextSample();

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
        bool ignoreMissingGraphemes = util::KnobDatabase::getKnobValue(
            "InputAudioDataProducer::IgnoreMissingGraphemes", false);

        if(label.empty())
        {
            return StringVector();
        }

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
                if(ignoreMissingGraphemes)
                {
                    remainingLabel = remainingLabel.substr(1);
                    continue;
                }
                else
                {
                    throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                        "' against a grapheme.");
                }
            }

            --insertPosition;

            auto grapheme = remainingLabel.substr(0, insertPosition->size());

            if(grapheme != *insertPosition)
            {
                if(ignoreMissingGraphemes)
                {
                    remainingLabel = remainingLabel.substr(1);
                    continue;
                }
                else
                {
                    throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                        "' against best grapheme '" + *insertPosition + "'.");
                }
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
        if(_databaseFilename.empty())
        {
            util::log("InputAudioDataProducer") << " audio database not specified, skipping.\n";
            return;
        }

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
                if(!Audio(sample.path()).isValid())
                {
                    util::log("InputAudioDataProducer::Detail") << "  found invalid audio sample '"
                        << sample.path() << "'\n";
                    continue;
                }
                else if(sample.hasLabel())
                {
                    util::log("InputAudioDataProducer::Detail") << "  found labeled audio '"
                        << sample.path() << "' with label '" << sample.label() << "'\n";
                }
                else
                {
                    util::log("InputAudioDataProducer::Detail") << "  found unlabeled audio '"
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
    typedef std::set<size_t> CacheSet;

    class AudioDescriptor
    {
    public:
        enum Type
        {
            AudioType,
            NoiseType,
            RepeatedType,
            InvalidType
        };

    public:
        AudioDescriptor(Type type, size_t audioVectorOffset, size_t audioClipDuration)
        : type(type), audioVectorOffset(audioVectorOffset), audioClipDuration(audioClipDuration)
        {

        }

    public:
        Type   type;
        size_t audioVectorOffset;
        size_t audioClipDuration;
    };

    class Sample
    {
    public:
        typedef AudioDescriptor::Type Type;

    public:
        Sample(Type type, size_t audioVectorOffset, size_t audioClipOffset,
            size_t audioClipDuration)
        : type(type), audioVectorOffset(audioVectorOffset), audioClipOffset(audioClipOffset),
          audioClipDuration(audioClipDuration)
        {

        }

        Sample()
        : Sample(AudioDescriptor::InvalidType, 0, 0, 0)
        {

        }

    public:
        Type   type;
        size_t audioVectorOffset;
        size_t audioClipOffset;
        size_t audioClipDuration;

    public:
        CacheSet cacheSet;
    };

    typedef std::vector<Sample> SampleVector;
    typedef std::vector<AudioDescriptor> AudioDescriptorVector;

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

            // Determine how many audio samples to cache
            _secondsToCache = util::KnobDatabase::getKnobValue(
                "InputAudioDataProducer::SecondsToCache", 3600);

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

            if(sample.type == AudioDescriptor::AudioType)
            {
                result = _audio[sample.audioVectorOffset];
            }
            else if(sample.type == AudioDescriptor::NoiseType)
            {
                size_t end = (sample.audioClipOffset + _totalTimestepsPerNoise) *
                    _getFrameSize();
                result = _audio[sample.audioVectorOffset].slice(begin, end);
            }
            else
            {
                assert(sample.type == AudioDescriptor::RepeatedType);
                size_t end = (sample.audioClipOffset + _totalTimestepsPerRepeat) *
                    _getFrameSize();
                result = _audio[sample.audioVectorOffset].slice(begin, end);
            }

            _updateCache();

            result = result.powerNormalize();

            result.setUnpaddedLength(result.size());

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

            util::log("InputAudioDataProducer::Detail") << "Adding audio sample with frameCount "
                << frameCount << "\n";

            if(frameCount <= _totalTimestepsPerUtterance)
            {
                util::log("InputAudioDataProducer::Detail") << " Adding it with index " << index
                    << ".\n";
                _descriptors.push_back(AudioDescriptor(AudioDescriptor::AudioType,
                    index, frameCount));
            }
            else
            {
                util::log("InputAudioDataProducer::Detail") << " Skipping it because it is "
                    "longer than the max audio size '" << _totalTimestepsPerUtterance << "'.\n";
            }
        }

        void addNoise(const Audio& noise)
        {
            size_t index = _audio.size();

            _audio.push_back(noise);

            _audio.back().cacheHeader();

            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

            size_t frameCount = _audio.back().duration() * frequency / _getFrameSize();

            _descriptors.push_back(AudioDescriptor(AudioDescriptor::NoiseType,
                index, frameCount));
        }

        void addRepeatedAudio(const Audio& audio)
        {
            size_t index = _audio.size();

            _audio.push_back(audio);

            _audio.back().cacheHeader();

            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

            size_t frameCount = _audio.back().duration() * frequency / _getFrameSize();

            _descriptors.push_back(AudioDescriptor(AudioDescriptor::RepeatedType,
                index, frameCount));
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
            size_t remainingDescriptors = _getUniqueDescriptorCount();
                  _nextSample           = 0;

            size_t remainingSamples = getUniqueSampleCount();

            std::shuffle(_descriptors.begin(), _descriptors.begin() + remainingDescriptors,
                _generator);

            size_t sampleBegin = 0;
            size_t sampleOffsetInDescriptor = 0;

            for(size_t begin = 0; begin < remainingDescriptors; )
            {
                SampleVector sortedSamples = _getSortedSamplesForRange(begin,
                    sampleOffsetInDescriptor);

                size_t usableSamples = std::min(remainingSamples, sortedSamples.size());

                std::copy(sortedSamples.begin(), sortedSamples.begin() + usableSamples,
                    _samples.begin() + sampleBegin);

                sampleBegin += usableSamples;
                remainingSamples -= usableSamples;

                if(remainingSamples == 0)
                {
                    break;
                }
            }

            _updateCache();
        }

        void initialize()
        {
            _createSampleVector();

            reset();

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
        size_t _getUniqueDescriptorCount() const
        {
            size_t coveredSamples = 0;
            size_t enoughDescriptorsToCoverAllSamples = 0;

            for(auto& descriptor : _descriptors)
            {
                coveredSamples += _getSamplesPerDescriptor(descriptor);
                ++enoughDescriptorsToCoverAllSamples;

                if(coveredSamples >= getUniqueSampleCount())
                {
                    break;
                }
            }

            return enoughDescriptorsToCoverAllSamples;;
        }

    private:
        SampleVector _getSortedSamplesForRange(size_t& descriptorIndex,
            size_t& sampleOffsetInDescriptor)
        {
            SampleVector samples;

            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");
            size_t timestepsToCache = _secondsToCache * frequency / _getFrameSize();

            assert(timestepsToCache > 0);

            size_t timestepsSoFar = 0;

            CacheSet cachedFiles;

            size_t startingDescriptor = descriptorIndex;

            while(timestepsSoFar < timestepsToCache)
            {
                if(!_findValidSample(descriptorIndex, sampleOffsetInDescriptor))
                {
                    break;
                }

                samples.push_back(_createSample(descriptorIndex, sampleOffsetInDescriptor));

                cachedFiles.insert(samples.back().audioVectorOffset);
                timestepsSoFar += samples.back().audioClipDuration;
                sampleOffsetInDescriptor += samples.back().audioClipDuration;
            }

            util::log("InputAudioDataProducer") << "Creating cache segment with "
                << samples.size() << " samples and "
                << (descriptorIndex - startingDescriptor) << " files.\n";

            std::shuffle(samples.begin(), samples.end(), _generator);

            for(size_t start = 0; start < samples.size(); start += _randomWindow)
            {
                size_t end = std::min(samples.size(), start + _randomWindow);

                std::sort(samples.begin() + start, samples.begin() + end,
                    [](const Sample& left, const Sample& right)
                    {
                        return left.audioClipDuration < right.audioClipDuration;
                    });
            }

            for(auto& sample : samples)
            {
                sample.cacheSet = cachedFiles;
            }

            return samples;
        }

        bool _findValidSample(size_t& descriptorIndex, size_t& sampleOffsetInDescriptor)
        {
            while(true)
            {
                if(descriptorIndex >= _descriptors.size())
                {
                    return false;
                }

                size_t nextSampleSize = _getNextSampleSize(descriptorIndex);

                if(sampleOffsetInDescriptor + nextSampleSize <=
                    _descriptors[descriptorIndex].audioClipDuration)
                {
                    return true;
                }

                descriptorIndex += 1;
                sampleOffsetInDescriptor = 0;
            }
        }

        size_t _getNextSampleSize(size_t descriptorIndex) const
        {
            return _getNextSampleSize(_descriptors[descriptorIndex]);
        }

        size_t _getNextSampleSize(const AudioDescriptor& descriptor) const
        {
            if(descriptor.type == AudioDescriptor::AudioType)
            {
                return descriptor.audioClipDuration;
            }
            else if(descriptor.type == AudioDescriptor::RepeatedType)
            {
                return _totalTimestepsPerRepeat;
            }
            else
            {
                assert(descriptor.type == AudioDescriptor::NoiseType);
                return _totalTimestepsPerNoise;
            }
        }

        size_t _getSamplesPerDescriptor(const AudioDescriptor& descriptor) const
        {
            size_t sampleSize = _getNextSampleSize(descriptor);

            return descriptor.audioClipDuration / sampleSize;
        }

        Sample _createSample(size_t descriptorIndex, size_t sampleOffsetInDescriptor)
        {
            return Sample(_descriptors[descriptorIndex].type,
                _descriptors[descriptorIndex].audioVectorOffset,
                sampleOffsetInDescriptor, _getNextSampleSize(descriptorIndex));
        }

    private:
        void _createSampleVector()
        {
            size_t totalSamples = 0;

            for(auto& fileDescriptor : _descriptors)
            {
                totalSamples += _getSamplesInDescriptor(fileDescriptor);
            }

            _samples.resize(totalSamples);
        }

        size_t _getSamplesInDescriptor(const AudioDescriptor& descriptor) const
        {
            if(descriptor.type == AudioDescriptor::AudioType)
            {
                return 1;
            }
            else if(descriptor.type == AudioDescriptor::RepeatedType)
            {
                return descriptor.audioClipDuration / _totalTimestepsPerRepeat;
            }
            else
            {
                assert(descriptor.type == AudioDescriptor::NoiseType);
                return descriptor.audioClipDuration / _totalTimestepsPerNoise;
            }
        }

    private:

        void _updateCache()
        {
            _updateCache(_audio, _cachedAudio);
        }

        void _updateCache(AudioVector& audio, CacheSet& cacheSet)
        {
            CacheSet newCacheSet;

            if(!empty())
            {
                newCacheSet = _samples[_nextSample].cacheSet;
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
                    _downsample(audio[element]);

                }
            }

            cacheSet = std::move(newCacheSet);
        }

    private:
        void _downsample(Audio& audio)
        {
            size_t frequency = _producer->getModel()->getAttribute<size_t>("SamplingRate");

            if(audio.frequency() != frequency)
            {
                audio = audio.sample(frequency);
            }
        }

    private:
        size_t _getFrameSize() const
        {
            return _producer->getInputSize()[0];
        }

    private:
        InputAudioDataProducer* _producer;

    private:
        AudioVector _audio;

    private:
        AudioDescriptorVector _descriptors;
        SampleVector          _samples;

    private:
        CacheSet _cachedAudio;

    private:
        std::default_random_engine _generator;

    private:
        size_t _randomWindow;
        size_t _totalTimestepsPerUtterance;
        size_t _totalTimestepsPerRepeat;
        size_t _totalTimestepsPerNoise;
        double _secondsToCache;

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

void InputAudioDataProducer::addRawSample(const void* data, size_t size, const std::string& type,
    const std::string& label)
{
    _implementation->addRawSample(data, size, type, label);
}

}

}



