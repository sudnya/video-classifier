/*! \file   Sample.cpp
    \date   Saturday December 6, 2013
    \author Gregory Diamos <solusstutus@gmail.com>
    \brief  The source file for the Sample class.
*/

#include <lucius/database/interface/Sample.h>

#include <lucius/audio/interface/Audio.h>

#include <lucius/video/interface/Video.h>
#include <lucius/video/interface/Image.h>

namespace lucius
{

namespace database
{

Sample::Sample(const std::string& path,
    const std::string& label,
    size_t beginFrame, size_t endFrame)
: _path(path), _label(label),
  _beginFrame(beginFrame), _endFrame(endFrame)
{

}

const std::string& Sample::path() const
{
    return _path;
}

const std::string& Sample::label() const
{
    return _label;
}

size_t Sample::beginFrame() const
{
    return _beginFrame;
}

size_t Sample::endFrame() const
{
    return _endFrame;
}

bool Sample::isVideoSample() const
{
    return video::Video::isPathAVideoFile(path());
}

bool Sample::isImageSample() const
{
    return video::Image::isPathAnImageFile(path());
}

bool Sample::isAudioSample() const
{
    return audio::Audio::isPathAnAudioFile(path());
}

bool Sample::isTextSample() const
{
    assert(false);
    return false; 
}

bool Sample::hasLabel() const
{
    return !label().empty();
}

}

}


