/*    \file   WavAudioLibrary.cpp
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the WavAudioLibrary class.
*/

// Lucius Includes
#include <lucius/audio/interface/WavAudioLibrary.h>

// Standard Library Includes
#include <cassert>

namespace lucius
{

namespace audio
{

WavAudioLibrary::~WavAudioLibrary()
{

}

class ArrayWrapper
{
public:
    uint8_t array[4];
};

class RiffWavHeader
{
public:
    ArrayWrapper groupId;
    uint32_t     size;
    ArrayWrapper riffType;
};

class FormatHeader
{
public:
    ArrayWrapper id;
    uint32_t     size;

    int16_t  formatTag;
    uint16_t channels;
    uint32_t samplesPerSec;
    uint32_t avgBytesPerSec;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};

class DataHeader
{
public:
    ArrayWrapper id;
    int32_t      size;
};

class GenericHeader
{
public:
    ArrayWrapper id;
    int32_t      size;
};

static RiffWavHeader loadRiffHeader(size_t position, std::istream& stream)
{
    stream.seekg(position);

    RiffWavHeader header;

    stream.read(reinterpret_cast<char*>(&header), sizeof(RiffWavHeader));

    if(!stream.good())
    {
        throw std::runtime_error("Failed to load riff wave header.");
    }

    return header;
}

static std::string toString(const ArrayWrapper& id)
{
    std::string string;

    string.push_back(id.array[0]);
    string.push_back(id.array[1]);
    string.push_back(id.array[2]);
    string.push_back(id.array[3]);

    return string;
}

static void checkRiffHeader(const RiffWavHeader& header)
{
    if(toString(header.groupId) != "RIFF")
    {
        throw std::runtime_error("Header format for WAV is not correct, expecting RIFF, "
            "but got '" + toString(header.groupId) + "'");
    }

    if(toString(header.riffType) != "WAVE")
    {
        throw std::runtime_error("Header format for WAV is not correct, expecting WAVE, "
            "but got '" + toString(header.riffType) + "'");
    }
}

static void seekToHeader(const std::string& headerLabel, size_t position, std::istream& stream)
{
    stream.seekg(position + sizeof(RiffWavHeader), std::ios::beg);

    while(stream.good())
    {
        size_t checkpoint = stream.tellg();

        GenericHeader header;

        stream.read(reinterpret_cast<char*>(&header), sizeof(GenericHeader));

        if(toString(header.id) == headerLabel)
        {
            stream.seekg(checkpoint, std::ios::beg);

            return;
        }

        stream.seekg(header.size, std::ios::cur);
    }

    throw std::runtime_error("Could not find header with label '" + headerLabel + "'");
}

static FormatHeader loadFormatHeader(size_t position, std::istream& stream)
{
    seekToHeader("fmt ", position, stream);

    FormatHeader header;

    stream.read(reinterpret_cast<char*>(&header), sizeof(FormatHeader));

    if(!stream.good())
    {
        throw std::runtime_error("Failed to load format header.");
    }

    return header;
}

static DataHeader loadDataHeader(size_t position, std::istream& stream)
{
    seekToHeader("data", position, stream);

    DataHeader header;

    stream.read(reinterpret_cast<char*>(&header), sizeof(DataHeader));

    if(!stream.good())
    {
        throw std::runtime_error("Failed to load data header.");
    }

    return header;
}

WavAudioLibrary::HeaderAndData WavAudioLibrary::loadAudio(
    std::istream& stream, const std::string& format)
{
    assert(format == ".wav");

    auto basePosition = stream.tellg();

    auto riffHeader = loadRiffHeader(basePosition, stream);

    checkRiffHeader(riffHeader);

    auto formatHeader = loadFormatHeader(basePosition, stream);
    auto dataHeader   = loadDataHeader(basePosition, stream);

    HeaderAndData headerAndData;

    headerAndData.header.bytesPerSample = formatHeader.bitsPerSample / 8;

    size_t sampleSize = formatHeader.channels * headerAndData.header.bytesPerSample;

    headerAndData.header.samples      = dataHeader.size / (sampleSize);
    headerAndData.header.samplingRate = formatHeader.samplesPerSec;

    headerAndData.data.resize(headerAndData.header.bytesPerSample * headerAndData.header.samples);

    uint8_t sampleBuffer[sampleSize];

    for(size_t sample = 0; sample != headerAndData.header.samples; ++sample)
    {
        stream.read(reinterpret_cast<char*>(sampleBuffer), sampleSize);

        if(!stream.good())
        {
            throw std::runtime_error("Reading WAV data from stream failed.");
        }

        std::memcpy(&headerAndData.data[sample * headerAndData.header.bytesPerSample],
            sampleBuffer, headerAndData.header.bytesPerSample);
    }

    return headerAndData;
}

static ArrayWrapper toByte4Array(const std::string& string)
{
    ArrayWrapper array;

    array.array[0] = string[0];
    array.array[1] = string[1];
    array.array[2] = string[2];
    array.array[3] = string[3];

    return array;
}

static void writeRiffHeader(std::ostream& stream, size_t dataSize)
{
    RiffWavHeader header;

    header.groupId  = toByte4Array("RIFF");
    header.size     = 4 + sizeof(FormatHeader) + sizeof(DataHeader) + dataSize;
    header.riffType = toByte4Array("WAVE");

    stream.write(reinterpret_cast<char*>(&header), sizeof(RiffWavHeader));

    if(!stream.good())
    {
        throw std::runtime_error("Failed to write RIFF header.");
    }
}

static void writeFormatHeader(std::ostream& stream, const WavAudioLibrary::Header& audioHeader)
{
    FormatHeader header;

    header.id   = toByte4Array("fmt ");
    header.size = sizeof(FormatHeader) - 8;

    header.formatTag      = 1;
    header.channels       = 1;
    header.samplesPerSec  = audioHeader.samplingRate;
    header.avgBytesPerSec = audioHeader.samplingRate * audioHeader.bytesPerSample;
    header.blockAlign     = audioHeader.bytesPerSample;
    header.bitsPerSample  = 8 * audioHeader.bytesPerSample;

    stream.write(reinterpret_cast<char*>(&header), sizeof(FormatHeader));

    if(!stream.good())
    {
        throw std::runtime_error("Failed to write WAV format header.");
    }
}

static void writeData(std::ostream& stream, const WavAudioLibrary::Header& header,
    const WavAudioLibrary::DataVector& data)
{
    DataHeader dataHeader;

    dataHeader.id   = toByte4Array("data");
    dataHeader.size = data.size();

    stream.write(reinterpret_cast<char*>(&dataHeader), sizeof(DataHeader));

    if(!stream.good())
    {
        throw std::runtime_error("Failed to write WAV data header.");
    }

    stream.write(reinterpret_cast<const char*>(data.data()), data.size());

}

void WavAudioLibrary::saveAudio(std::ostream& stream, const std::string& format,
    const Header& header, const DataVector& data)
{
    assert(format == ".wav");

    writeRiffHeader(stream, data.size());

    writeFormatHeader(stream, header);
    writeData(stream, header, data);
}

WavAudioLibrary::StringVector WavAudioLibrary::getSupportedExtensions() const
{
    return StringVector({".wav"});
}

}

}


