/*! \file   Sample.cpp
	\date   Saturday December 6, 2013
	\author Gregory Diamos <solusstutus@gmail.com>
	\brief  The source file for the Sample class.
*/

#include <lucious/database/interface/Sample.h>

#include <lucious/video/interface/Video.h>

namespace lucious
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
	return video::Video::isPathAVideo(path());
}

bool Sample::isImageSample() const
{
	return video::Image::isPathAnImage(path());
}

bool Sample::hasLabel() const
{
	return !label().empty();
}

}

}


