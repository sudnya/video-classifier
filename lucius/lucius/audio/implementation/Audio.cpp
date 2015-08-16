/*! \file   Audio.cpp
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Audio class.
*/

// Standard Library Includes
#include <string>
#include <map>

namespace lucius
{

namespace audio
{

Audio::Audio(const std::string& path, const std::string& label)
: _path(path), _isLoaded(path), _defaultLabel(label)
{

}

Audio::Audio(size_t samples, size_t bytesPerSample, size_t frequency)
: _isLoaded(true), _samples(samples), _bytesPerSample(bytesPerSample), _samplingRate(frequency)
{

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

    std::swap(_data, ByteVector());
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

size_t Audio::resize(size_t timesteps)
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
        return reinterpret_cast<int8_t&>(_data[timestep]);
    }
    case 2:
    {
        return reinterpret_cast<int16_t&>(_data[timestep]);
    }
    case 4:
    {
        return reinterpret_cast<int32_t&>(_data[timestep]);
    }
    case 8:
    {
        return reinterpret_cast<int64_t&>(_data[timestep]);
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
    }
    case 2:
    {
        reinterpret_cast<int16_t&>(_data[timestep]) value;
    }
    case 4:
    {
        reinterpret_cast<int32_t&>(_data[timestep]) value;
    }
    case 8:
    {
        reinterpret_cast<int64_t&>(_data[timestep]) = value;
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

void* Audio::getDataForTimestep(size_t timestep)
{
    return &_data[timestep * bytesPerSample()];
}

const void* Audio::getDataForTimestep(size_t timestep) const
{
    return &_data[timestep * bytesPerSample()];
}

void Audio::_load()
{
    if(_isLoaded)
    {
        return;
    }

    auto header = AudioLibraryInterface::loadHeader(_path);

    _samples        = header.samples;
    _bytesPerSample = header.bytesPerSample;
    _samplingRate   = header.samplingRate;

    _data = AudioLibraryInterface::loadData(_path);

    _isLoaded = true;
}

}

}


