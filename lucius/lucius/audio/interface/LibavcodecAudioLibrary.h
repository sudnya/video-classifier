/*	\file   LibavcodecAudioLibrary.h
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LibavcodecAudioLibrary class.
*/

#pragma once

// Lucius Includes
#include <lucius/video/interface/AudioLibraryInterface.h>

namespace lucius
{

namespace audio
{

class LibavcodecAudioLibrary
{
public:
	typedef AudioLibraryInterface::Header     Header;
	typedef AudioLibraryInterface::DataVector DataVector;
	typedef std::vector<std::string>          StringVector;

public:
	virtual ~LibavcodecAudioLibrary();

public:
	virtual Header     loadHeader(const std::string& path);
	virtual DataVector loadData  (const std::string& path);

public:
	virtual void saveAudio(const std::string& path, const Header& header,
		const DataVector& data);

public:
	virtual StringVector getSupportedExtensions() const;


};

}

}





