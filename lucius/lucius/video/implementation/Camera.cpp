/*	\file   Camera.cpp
	\date   Sunday August 11, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Camera class.
*/

// Lucius Includes
#include <lucius/video/interface/Camera.h>

#include <lucius/video/interface/VideoLibrary.h>
#include <lucius/video/interface/VideoStream.h>
#include <lucius/video/interface/VideoLibraryInterface.h>

#include <lucius/video/interface/Image.h>

namespace lucius
{

namespace video
{

Camera::Camera()
: _library(VideoLibraryInterface::getLibraryThatSupportsCamera())
{
    if(_library)
    {
        _stream.reset(_library->newCameraStream());
    }
}

Camera::~Camera()
{

}

Image Camera::popFrame()
{
    Image nextFrame;

    _stream->getNextFrame(nextFrame);

    return nextFrame;
}

}

}


