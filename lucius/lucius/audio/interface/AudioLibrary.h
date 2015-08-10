/*	\file   AudioLibrary.h
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the AudioLibrary class.
*/

#pragma once

// Lucius Includes
#include <lucius/video/interface/AudioLibraryInterface.h>

namespace lucius
{

namespace audio
{

class AudioLibrary
{
public:
	typedef AudioLibraryInterface::Header     Header;
	typedef AudioLibraryInterface::DataVector DataVector;
	typedef std::vector<std::string>          StringVector;

public:
	virtual ~AudioLibrary();

public:
	virtual Header     loadHeader(const std::string& path) = 0;
	virtual DataVector loadData  (const std::string& path) = 0;

public:
	virtual void saveAudio(const std::string& path, const Header& header,
		const DataVector& data) = 0;

public:
	virtual StringVector getSupportedExtensions() const = 0;


};

}

}




