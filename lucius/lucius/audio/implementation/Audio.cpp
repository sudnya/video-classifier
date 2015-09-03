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

namespace lucius
{

namespace audio
{

Audio::Audio(const std::string& path, const std::string& label)
: _path(path), _isLoaded(false), _defaultLabel(label)
{

}

Audio::Audio(std::istream& stream, const std::string& format)
: _isLoaded(false)
{
    _load(stream, format);
}

Audio::Audio(size_t samples, size_t bytesPerSample, size_t frequency)
: _isLoaded(true), _samples(samples), _bytesPerSample(bytesPerSample), _samplingRate(frequency)
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

void Audio::clearLabels()
{
    _labels.clear();
}

void Audio::cache()
{
    _load();
}

void Audio::invalidateCache()
{
    _isLoaded = false;

   _data = std::move(ByteVector());
}

bool Audio::isCached() const
{
    return _isLoaded;
}

size_t Audio::frequency() const
{
    assert(_isLoaded);

    return _samplingRate;
}

double Audio::duration() const
{
    assert(_isLoaded);

    return (size() + 0.0) / _samplingRate;
}

size_t Audio::size() const
{
    assert(_isLoaded);

    return _samples;
}

size_t Audio::bytes() const
{
    assert(_isLoaded);

    return _data.size();
}

size_t Audio::bytesPerSample() const
{
    assert(_isLoaded);

    return _bytesPerSample;
}

void Audio::resize(size_t timesteps)
{
    _load();

    _samples = timesteps;
    _data.resize(_samples * _bytesPerSample);
}

double Audio::getSample(size_t timestep) const
{
    assert(_isLoaded);

    switch(_bytesPerSample)
    {
    case 1:
    {
        return reinterpret_cast<const int8_t&>(_data[timestep]);
    }
    case 2:
    {
        return reinterpret_cast<const int16_t&>(_data[timestep]);
    }
    case 4:
    {
        return reinterpret_cast<const int32_t&>(_data[timestep]);
    }
    case 8:
    {
        return reinterpret_cast<const int64_t&>(_data[timestep]);
    }
    default:
        break;
    }

    assertM(false, "Invalid bytes per sample.");
}

void Audio::setSample(size_t timestep, double value)
{
    switch(_bytesPerSample)
    {
    case 1:
    {
        reinterpret_cast<int8_t&>(_data[timestep]) = value;
        break;
    }
    case 2:
    {
        reinterpret_cast<int16_t&>(_data[timestep]) = value;
        break;
    }
    case 4:
    {
        reinterpret_cast<int32_t&>(_data[timestep]) = value;
        break;
    }
    case 8:
    {
        reinterpret_cast<int64_t&>(_data[timestep]) = value;
        break;
    }
    default:
    {
        assertM(false, "Invalid bytes per sample.");
    }
    }
}

Audio Audio::slice(size_t startingTimestep, size_t endingTimestep) const
{
    Audio result(endingTimestep - startingTimestep, bytesPerSample(), frequency());

    std::memcpy(result.getDataForTimestep(0), getDataForTimestep(startingTimestep),
        result.bytes());

    return result;
}

std::string Audio::getLabelForTimestep(size_t timestep) const
{
    auto label = _labels.lower_bound(timestep);

    if(label != _labels.end() && timestep < label->second.endTimestep)
    {
        return label->second.label;
    }

    return _defaultLabel;
}

std::string Audio::label() const
{
    return _defaultLabel;
}

void* Audio::getDataForTimestep(size_t timestep)
{
    return &_data[timestep * bytesPerSample()];
}

const void* Audio::getDataForTimestep(size_t timestep) const
{
    return &_data[timestep * bytesPerSample()];
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
: startTimestep(start), endTimestep(end), label(l)
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
}

}

}


