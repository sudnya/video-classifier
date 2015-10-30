/*    \file   OpenCVVideoLibrary.cpp
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the OpenCVVideoLibrary class.
*/

// Lucius Includes
#include <lucius/video/interface/OpenCVVideoLibrary.h>
#include <lucius/video/interface/OpenCVLibrary.h>
#include <lucius/video/interface/Image.h>

#include <lucius/util/interface/string.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/paths.h>

// Standard Library Includes
#include <stdexcept>
#include <cstring>

namespace lucius
{

namespace video
{


VideoStream* OpenCVVideoLibrary::newStream(const std::string& path)
{
    return new OpenCVVideoStream(path);
}

VideoStream* OpenCVVideoLibrary::newCameraStream()
{
    return new OpenCVVideoStream();
}

void OpenCVVideoLibrary::freeStream(VideoStream* s)
{
    delete s;
}

OpenCVVideoLibrary::StringVector
    OpenCVVideoLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".avi|.mp4|camera", "|"));
}

OpenCVVideoLibrary::OpenCVVideoStream::OpenCVVideoStream(
    const std::string& path)
: _path(path), _cvCapture(OpenCVLibrary::cvCreateFileCapture(path.c_str()))
{

}

OpenCVVideoLibrary::OpenCVVideoStream::OpenCVVideoStream()
: _cvCapture(OpenCVLibrary::cvCaptureFromCAM(0))
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

    int colorSize = (image->depth % 32) / 8;
    int pixelSize = image->nChannels * colorSize;

    Image::ByteVector data(image->height * image->width *
        pixelSize);

    for(int y = 0; y < image->height; ++y)
    {
        int position = y * image->widthStep;
        int dataPosition = y * image->width * pixelSize;

        std::memcpy(&data[dataPosition], &image->imageData[position],
            std::min(image->widthStep, image->width * pixelSize));
    }

    frame = Image(image->width, image->height, image->nChannels,
        colorSize, _path, "", data);

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


