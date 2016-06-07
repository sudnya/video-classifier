/*    \file   AudioLibraryInterface.h
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the AudioLibraryInterface class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>
#include <cstdint>
#include <istream>
#include <ostream>

namespace lucius
{

namespace audio
{

class AudioLibraryInterface
{
public:
    class Header
    {
    public:
        Header(size_t _samples, size_t _bytesPerSample, size_t _samplingRate);
        Header();

    public:
        size_t samples;
        size_t bytesPerSample;
        size_t samplingRate;

    public:
        bool operator==(const Header&) const;
    };

public:
    typedef std::vector<uint8_t> DataVector;

public:
    class HeaderAndData
    {
    public:
        Header     header;
        DataVector data;
    };

public:
    static bool isAudioTypeSupported(const std::string& extension);

public:
    static HeaderAndData loadAudio(std::istream& path, const std::string& format);
    static Header loadAudioHeader(std::istream& path, const std::string& format);

public:
    static void saveAudio(std::ostream& path, const std::string& format,
        const Header& header, const DataVector& data);

};

}

}



