/*	\file   Video.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Video class.
*/

// Minerva Includes
#include <minerva/video/interface/Video.h>
#include <minerva/video/interface/VideoLibraryInterface.h>
#include <minerva/video/interface/VideoLibrary.h>
#include <minerva/video/interface/VideoStream.h>

#include <minerva/util/interface/paths.h>

// Standard Library Includes
#include <stdexcept>
#include <sstream>

namespace minerva
{

namespace video
{

Video::Video(const std::string& p)
: _path(p), _frame(0), _library(nullptr), _stream(nullptr)
{

}

Video::~Video()
{
	invalidateCache();
}

Video::Video(const Video& v)
: _path(v._path), _frame(v._frame), _library(nullptr), _stream(nullptr)
{

}

Video& Video::operator=(const Video& v)
{
	invalidateCache();
	
	_path  = v._path;
	_frame = v._frame;
	
	return *this;
}

bool Video::loaded() const
{
	return _library != nullptr;
}

void Video::load()
{
	if(loaded()) return;
	
	_library = VideoLibraryInterface::getLibraryThatSupports(_path);
	
	if(!loaded())
	{
		throw std::runtime_error("No video library can support '" +
			_path + "'");
	}
	
	_stream = _library->newStream(_path);
	
	for(unsigned int i = 0; i < _frame; ++i)
	{
		if(_stream->finished())
		{
			throw std::runtime_error("Could not seek to previously saved "
				"frame in video '" + _path + "'.");
		}
		
        Image image;

		_stream->getNextFrame(image);
	}
}

void Video::invalidateCache()
{
	if(_stream != nullptr)
	{
		_library->freeStream(_stream);
	}
	
	_library = nullptr;
	_stream  = nullptr;
}

ImageVector Video::getNextFrames(unsigned int frames)
{
	load();
	
	ImageVector images;
	
	for(unsigned int i = 0; i < frames; ++i)
	{
		if(_stream->finished()) break;

        Image image;
        
        if(!_stream->getNextFrame(image)) break;
		
        images.push_back(image);

		std::stringstream name;
		
		name << images.back().path() << "-frame-" << _frame;
	
		images.back().setPath(name.str());

		++_frame;
	}
	
	return images;
}

bool Video::finished()
{
	load();
	
	return _stream->finished();
}

bool Video::isPathAVideo(const std::string& path)
{
	auto extension = util::getExtension(path);
	
	return VideoLibraryInterface::isVideoTypeSupported(extension);
}

void Video::_seek(unsigned int frame)
{
	if(frame < _frame)
	{
		invalidateCache();
		_frame = 0;
	}
	
	while(frame > _frame)
	{
		Image i;
        
        _stream->getNextFrame(i);
		++_frame;
	}
}

}

}


