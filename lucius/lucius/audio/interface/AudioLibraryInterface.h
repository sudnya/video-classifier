/*	\file   AudioLibraryInterface.h
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the AudioLibraryInterface class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>
#include <cstdint>

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

	public:
        size_t samples;
        size_t bytesPerSample;
        size_t samplingRate;
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
	static HeaderAndData loadHeader(const std::string& path);

public:
	static void saveAudio(const std::string& path, const Header& header,
		const DataVector& data);

};

}

}



