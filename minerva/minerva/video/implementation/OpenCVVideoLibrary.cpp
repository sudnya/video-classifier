/*	\file   OpenCVVideoLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the OpenCVVideoLibrary class.
*/

// Minerva Includes
#include <minerva/video/interface/OpenCVVideoLibrary.h>
#include <minerva/video/interface/OpenCVLibrary.h>
#include <minerva/video/interface/Image.h>

#include <minerva/util/interface/string.h>
#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/paths.h>

// Standard Library Includes
#include <stdexcept>
#include <cstring>

namespace minerva
{

namespace video
{


VideoStream* OpenCVVideoLibrary::newStream(const std::string& path)
{
	return new OpenCVVideoStream(path);
}

void OpenCVVideoLibrary::freeStream(VideoStream* s)
{
	delete s;
}

OpenCVVideoLibrary::StringVector
	OpenCVVideoLibrary::getSupportedExtensions() const
{
	return StringVector(util::split(".avi|.mp4", "|"));
}

OpenCVVideoLibrary::OpenCVVideoStream::OpenCVVideoStream(
	const std::string& path)
: _path(path), _cvCapture(OpenCVLibrary::cvCreateFileCapture(path.c_str()))
{

}

OpenCVVideoLibrary::OpenCVVideoStream::~OpenCVVideoStream()
{
	OpenCVLibrary::cvReleaseCapture(&_cvCapture);
}

bool OpenCVVideoLibrary::OpenCVVideoStream::finished() const
{
	return OpenCVLibrary::cvGetCaptureProperty(_cvCapture,
		OpenCVLibrary::CV_CAP_PROP_POS_AVI_RATIO) == 1.0f;
}

bool OpenCVVideoLibrary::OpenCVVideoStream::getNextFrame(Image& frame)
{
	if(!OpenCVLibrary::cvGrabFrame(_cvCapture))
	{
		return false;
	}
	
	OpenCVLibrary::Image* image = OpenCVLibrary::cvRetrieveFrame(_cvCapture);

	if(image == nullptr)
	{
		throw std::runtime_error("Failed to retrieve frame from video stream.");
	}
	
	int pixelSize = (image->depth % 32) / 8;
	
	Image::ByteVector data(image->height * image->width *
		image->nChannels * pixelSize);
	
	for(int y = 0; y < image->height; ++y)
	{
		int position = y * image->widthStep;
		int dataPosition = y * image->width * image->nChannels * pixelSize;
		
		std::memcpy(&data[dataPosition], &image->imageData[position],
			image->widthStep);
	}
	
	frame = Image(image->width, image->height, image->nChannels,
		pixelSize, _path, data);

    return true;
}

size_t OpenCVVideoLibrary::OpenCVVideoStream::getTotalFrames() const
{
	return OpenCVLibrary::cvGetCaptureProperty(_cvCapture,
		OpenCVLibrary::CV_CAP_PROP_FRAME_COUNT);
}

bool OpenCVVideoLibrary::OpenCVVideoStream::seek(size_t frame)
{
	OpenCVLibrary::cvSetCaptureProperty(_cvCapture,
		OpenCVLibrary::CV_CAP_PROP_POS_FRAMES, frame);

	// TODO check for failure

	return true;
}

}

}


