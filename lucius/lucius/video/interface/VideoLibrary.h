/*	\file   VideoLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the VideoLibrary class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>

// Forward Declarations
namespace lucius { namespace video { class VideoStream; } }

namespace lucius
{

namespace video
{

class VideoLibrary
{
public:
	typedef std::vector<std::string> StringVector;

public:
	virtual ~VideoLibrary();

public:
	virtual VideoStream* newStream(const std::string& path) = 0;
	virtual VideoStream* newCameraStream() = 0;
	virtual void freeStream(VideoStream* s) = 0;

public:
	virtual StringVector getSupportedExtensions() const = 0;

};

}

}


