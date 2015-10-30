/*    \file   Camera.h
    \date   Sunday August 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Camera class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace video { class VideoLibrary; } }
namespace lucius { namespace video { class VideoStream;  } }
namespace lucius { namespace video { class Image;        } }

namespace lucius
{

namespace video
{

/*! \brief An interface to a camera on the computer. */
class Camera
{
public:
    Camera();
    ~Camera();

public:
    Image popFrame();

public:
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

private:
    std::unique_ptr<VideoLibrary> _library;
    std::unique_ptr<VideoStream>  _stream;

};

}

}

