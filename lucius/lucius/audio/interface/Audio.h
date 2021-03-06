/*! \file   Audio.h
    \date   Tuesday August 4, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Audio class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <map>
#include <vector>

namespace lucius
{

namespace audio
{

/*! \brief An abstraction for an audio sample. */
class Audio
{
public:
    Audio(const std::string& path, const std::string& defaultLabel = "");
    Audio(std::istream& stream, const std::string& format, const std::string& defaultLabel);

    Audio(size_t samples, size_t bytesPerSample, size_t frequency, const std::string& path = "");
    Audio();

public:
    void save(std::ostream& stream, const std::string& format);

public:
    void addLabel(size_t start, size_t end, const std::string& label);
    void setDefaultLabel(const std::string& label);

public:
    void clearLabels();

public:
    void cache();
    void cacheHeader();
    void invalidateCache();

    bool isCached() const;
    bool isValid();

public:
    size_t frequency() const;

public:
    double duration() const;
    size_t size() const;
    size_t bytes() const;
    size_t bytesPerSample() const;

public:
    const std::string& path() const;

public:
    void resize(size_t samples);

public:
    double getSample(size_t sample) const;
    void   setSample(size_t sample, double value);

public:
    void setUnpaddedLength(size_t position);
    size_t getUnpaddedLength() const;

public:
    Audio slice(size_t startingSample, size_t endingSample) const;
    Audio sample(size_t newFrequency) const;
    Audio powerNormalize() const;

public:
    std::string getLabelForSample(size_t sample) const;

public:
    std::string label() const;

public:
          void* getDataForSample(size_t sample);
    const void* getDataForSample(size_t sample) const;

public:
    bool operator==(const Audio& ) const;
    bool operator!=(const Audio& ) const;

public:
    static bool isPathAnAudioFile(const std::string& );

private:
    std::string _path;

private:
    bool _isLoaded;
    bool _isHeaderLoaded;

private:
    class Label
    {
    public:
        Label(size_t start, size_t end, const std::string& label);

    public:
        size_t startSample;
        size_t endSample;

    public:
        std::string label;
    };

private:
    typedef std::map<size_t, Label> LabelMap;

private:
    typedef std::vector<uint8_t> ByteVector;

private:
    LabelMap _labels;

    std::string _defaultLabel;

    size_t _unpaddedSequenceLength;

private:
    size_t _samples;
    size_t _bytesPerSample;
    size_t _samplingRate;

private:
    ByteVector _data;

private:
    void _load();
    void _loadHeader();
    void _load(std::istream& stream, const std::string& format);

};

}

}

