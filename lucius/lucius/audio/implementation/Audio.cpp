/*! \file   Audio.cpp
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Audio class.
*/

// Lucius Includes
#include <lucius/audio/interface/Audio.h>
#include <lucius/audio/interface/AudioLibraryInterface.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/paths.h>

// Standard Library Includes
#include <cassert>
#include <fstream>
#include <cstring>
#include <cmath>

namespace lucius
{

namespace audio
{

Audio::Audio(const std::string& path, const std::string& label)
: _path(path), _isLoaded(false), _isHeaderLoaded(false),
  _defaultLabel(label), _unpaddedSequenceLength(0)
{

}

Audio::Audio(std::istream& stream, const std::string& format)
: _isLoaded(false), _isHeaderLoaded(false), _unpaddedSequenceLength(0)
{
    _load(stream, format);
}

Audio::Audio(size_t samples, size_t bytesPerSample, size_t frequency)
: _isLoaded(true), _isHeaderLoaded(true), _unpaddedSequenceLength(0),
  _samples(samples), _bytesPerSample(bytesPerSample), _samplingRate(frequency)
{
    _data.resize(_samples * _bytesPerSample);
}

Audio::Audio()
: Audio(0, 0, 0)
{
    _data.resize(_samples * _bytesPerSample);
}

void Audio::save(std::ostream& stream, const std::string& format)
{
    AudioLibraryInterface::saveAudio(stream, format,
        AudioLibraryInterface::Header(_samples, _bytesPerSample, _samplingRate), _data);
}

void Audio::addLabel(size_t start, size_t end, const std::string& label)
{
    _labels.emplace(start, Label(start, end, label));
}

void Audio::setDefaultLabel(const std::string& label)
{
    _defaultLabel = label;
}

void Audio::clearLabels()
{
    _labels.clear();
}

void Audio::cache()
{
    _load();
}

void Audio::cacheHeader()
{
    _loadHeader();
}

void Audio::invalidateCache()
{
    _isLoaded = false;
    _isHeaderLoaded = false;

   _data = ByteVector();
}

bool Audio::isCached() const
{
    return _isLoaded;
}

size_t Audio::frequency() const
{
    assert(_isHeaderLoaded);

    return _samplingRate;
}

double Audio::duration() const
{
    assert(_isHeaderLoaded);

    return (size() + 0.0) / _samplingRate;
}

size_t Audio::size() const
{
    assert(_isHeaderLoaded);

    return _samples;
}

size_t Audio::bytes() const
{
    assert(_isLoaded);

    return _data.size();
}

size_t Audio::bytesPerSample() const
{
    assert(_isHeaderLoaded);

    return _bytesPerSample;
}

void Audio::resize(size_t samples)
{
    _load();

    _samples = samples;
    _data.resize(_samples * _bytesPerSample);
}

double Audio::getSample(size_t sample) const
{
    assert(_isLoaded);

    switch(_bytesPerSample)
    {
    case 1:
    {
        return reinterpret_cast<const int8_t&>(_data[sample]);
    }
    case 2:
    {
        return reinterpret_cast<const int16_t&>(_data[sample * 2]);
    }
    case 4:
    {
        return reinterpret_cast<const int32_t&>(_data[sample * 4]);
    }
    case 8:
    {
        return reinterpret_cast<const int64_t&>(_data[sample * 8]);
    }
    default:
        break;
    }

    assertM(false, "Invalid bytes per sample.");
}

void Audio::setSample(size_t sample, double value)
{
    assert(_isLoaded);

    switch(_bytesPerSample)
    {
    case 1:
    {
        reinterpret_cast<int8_t&>(_data[sample]) = value;
        break;
    }
    case 2:
    {
        reinterpret_cast<int16_t&>(_data[sample * 2]) = value;
        break;
    }
    case 4:
    {
        reinterpret_cast<int32_t&>(_data[sample * 4]) = value;
        break;
    }
    case 8:
    {
        reinterpret_cast<int64_t&>(_data[sample * 8]) = value;
        break;
    }
    default:
    {
        assertM(false, "Invalid bytes per sample.");
    }
    }
}

void Audio::setUnpaddedLength(size_t position)
{
    _unpaddedSequenceLength = position;
}

size_t Audio::getUnpaddedLength() const
{
    return _unpaddedSequenceLength;
}

Audio Audio::slice(size_t startingSample, size_t endingSample) const
{
    Audio result(endingSample - startingSample, bytesPerSample(), frequency());

    std::memcpy(result.getDataForSample(0), getDataForSample(startingSample),
        result.bytes());

    result.setDefaultLabel(_defaultLabel);

    return result;
}

Audio Audio::sample(size_t newFrequency) const
{
    size_t samples = size() * newFrequency / frequency();

    Audio result(samples, bytesPerSample(), newFrequency);

    for(size_t sample = 0; sample != samples; ++sample)
    {
        size_t originalSample = std::min(sample * frequency() / newFrequency, size());

        result.setSample(sample, getSample(originalSample));
    }

    result.setDefaultLabel(_defaultLabel);

    return result;
}

Audio Audio::powerNormalize() const
{
    Audio result(size(), bytesPerSample(), frequency());

    size_t samples = size();

    double mean = 0.0;
    double sumOfSquaresOfDifferences = 0.0;
    double currentSamples = 0.0;

    for(size_t sample = 0; sample < samples; ++sample)
    {
        double value = getSample(sample);

        currentSamples += 1.0;

        double delta = value - mean;

        mean += delta / currentSamples;

        sumOfSquaresOfDifferences += delta * (value - mean);
    }

    double standardDeviation = std::sqrt(sumOfSquaresOfDifferences / (currentSamples - 1.0));

    for(size_t sample = 0; sample < samples; ++sample)
    {
        double value = getSample(sample);

        result.setSample(sample, (value - mean) / standardDeviation);
    }

    result.setDefaultLabel(_defaultLabel);

    return result;
}

std::string Audio::getLabelForSample(size_t samplePosition) const
{
    auto possibleLabel = _labels.lower_bound(samplePosition);

    if(possibleLabel != _labels.end() &&
        samplePosition >= possibleLabel->first &&
        samplePosition < possibleLabel->second.endSample)
    {
        return possibleLabel->second.label;
    }

    if(possibleLabel != _labels.begin())
    {
        --possibleLabel;
    }

    if(possibleLabel != _labels.end() &&
        samplePosition >= possibleLabel->first &&
        samplePosition < possibleLabel->second.endSample)
    {
        return possibleLabel->second.label;
    }

    return _defaultLabel;
}

std::string Audio::label() const
{
    return _defaultLabel;
}

void* Audio::getDataForSample(size_t sample)
{
    return &_data[sample * bytesPerSample()];
}

const void* Audio::getDataForSample(size_t sample) const
{
    return &_data[sample * bytesPerSample()];
}

bool Audio::operator==(const Audio& audio) const
{
    return  (_samples        == audio._samples) &&
            (_bytesPerSample == audio._bytesPerSample) &&
            (_samplingRate   == audio._samplingRate) &&
            (_data           == audio._data);
}

bool Audio::operator!=(const Audio& audio) const
{
    return !(*this == audio);
}

bool Audio::isPathAnAudio(const std::string& path)
{
    auto extension = util::getExtension(path);

    return AudioLibraryInterface::isAudioTypeSupported(extension);
}

Audio::Label::Label(size_t start, size_t end, const std::string& l)
: startSample(start), endSample(end), label(l)
{

}

void Audio::_load()
{
    if(_isLoaded)
    {
        return;
    }

    std::ifstream file(_path);

    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open " + _path + " for reading.");
    }

    util::log("Audio") << "Loading audio from '" + _path + "'\n";

    _load(file, util::getExtension(_path));
}

void Audio::_load(std::istream& stream, const std::string& format)
{
    if(_isLoaded)
    {
        return;
    }

    auto headerAndData = AudioLibraryInterface::loadAudio(stream, format);

    _samples        = headerAndData.header.samples;
    _bytesPerSample = headerAndData.header.bytesPerSample;
    _samplingRate   = headerAndData.header.samplingRate;

    _data = std::move(headerAndData.data);

    _isLoaded = true;
    _isHeaderLoaded = true;
}

void Audio::_loadHeader()
{
    if(_isHeaderLoaded)
    {
        return;
    }

    std::ifstream file(_path);

    util::log("Audio") << "Loading audio header from '" + _path + "'\n";

    if(!file.is_open())
    {
        throw std::runtime_error("Failed to open " + _path + " for reading.");
    }

    auto header = AudioLibraryInterface::loadAudioHeader(file, util::getExtension(_path));

    _samples        = header.samples;
    _bytesPerSample = header.bytesPerSample;
    _samplingRate   = header.samplingRate;

    _isHeaderLoaded = true;
}



}

}


