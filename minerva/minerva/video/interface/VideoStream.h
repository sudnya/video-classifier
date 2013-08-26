/*	\file   VideoStream.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the VideoStream class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace video { class Image; } }

namespace minerva
{

namespace video
{

class VideoStream
{	
public:
	virtual ~VideoStream();
	
public:
	virtual bool finished() const = 0;
	virtual bool getNextFrame(Image&) = 0;

};

}

}


