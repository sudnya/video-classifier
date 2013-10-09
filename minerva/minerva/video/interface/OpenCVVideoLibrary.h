/*	\file   OpenCVVideoLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the OpenCVVideoLibrary class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/VideoLibrary.h>
#include <minerva/video/interface/VideoStream.h>

// Forward Declarations
namespace minerva { namespace video { class CvCapture; } }

namespace minerva
{

namespace video
{

class OpenCVVideoLibrary : public VideoLibrary
{
public:
	virtual VideoStream* newStream(const std::string& path);
	virtual void freeStream(VideoStream* s);

public:
	virtual StringVector getSupportedExtensions() const;

public:
	class OpenCVVideoStream : public VideoStream
	{
	public:
		OpenCVVideoStream(const std::string& path);
		virtual ~OpenCVVideoStream();
		
	public:
		OpenCVVideoStream(const OpenCVVideoStream&) = delete;
		OpenCVVideoStream& operator=(const OpenCVVideoStream&) = delete;
		
	public:
		virtual bool finished() const;
		virtual bool getNextFrame(Image&);
		virtual size_t getTotalFrames() const;
		virtual bool seek(size_t frame);
	
	private:
		std::string _path;
		CvCapture*  _cvCapture;
	
	};

};

}

}


