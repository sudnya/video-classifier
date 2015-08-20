/*! \file   Audio.h
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Audio class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <map>

namespace lucius
{

namespace audio
{

/*! \brief An abstraction for an audio sample. */
class Audio
{
public:
    Audio(const std::string& path, const std::string& defaultLabel);
    Audio(std::istream& stream, const std::string& format);

    Audio(size_t samples, size_t bytesPerSample, size_t frequency)

public:
    void addLabel(size_t start, size_t end, const std::string& label);

public:
    void clearLabels();

public:
    void cache();
    void invalidateCache();

public:
    size_t frequency() const;

public:
    double duration() const;
    size_t size() const;
    size_t bytes() const;
    size_t bytesPerSample() const;

public:
    size_t resize(size_t timesteps);

public:
    double getSample(size_t timestep) const;
    void   setSample(size_t timestep, double value);

public:
    Audio slice(size_t startingTimestep, size_t endingTimestep) const;

public:
    std::string getLabelForTimestep(size_t timestep) const;

public:
          void* getDataForTimestep(size_t timestep);
    const void* getDataForTimestep(size_t timesetp) const;

public:
    bool operator==(const Audio& ) const;

private:
    std::string _path;

private:
    bool _isLoaded;

private:
    class Label
    {
    public:
        Label(size_t start, size_t end, const std::string& label);

    public:
        size_t startTimestep;
        size_t endTimestep

    public:
        std::string label;
    };

    typedef std::map<size_t, Label> LabelMap;

private:
    LabelMap _labels;

    std::string _defaultLabel;

private:
    size_t _samples;
    size_t _bytesPerSample;
    size_t _samplingRate;

private:
    ByteVector _data;

private:
    void _load();

};

}

}

