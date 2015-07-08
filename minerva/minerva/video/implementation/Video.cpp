/*	\file   Video.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Video class.
*/

// Lucious Includes
#include <lucious/video/interface/Video.h>
#include <lucious/video/interface/VideoLibraryInterface.h>
#include <lucious/video/interface/VideoLibrary.h>
#include <lucious/video/interface/VideoStream.h>

#include <lucious/util/interface/paths.h>
#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <stdexcept>
#include <sstream>

namespace lucious
{

namespace video
{

Video::Video(const std::string& p)
: _path(p), _frame(0), _library(nullptr), _stream(nullptr)
{

}

Video::Video(const std::string& path, const std::string& label,
	unsigned int beginFrame, unsigned int endFrame)
: _path(path), _frame(0), _labels(1, Label(label, beginFrame, endFrame)),
	_library(nullptr), _stream(nullptr)
{
	
}

Video::~Video()
{
	invalidateCache();
}

Video::Video(const Video& v)
: _path(v._path), _frame(v._frame), _labels(v._labels), _library(nullptr),
	_stream(nullptr)
{

}

Video& Video::operator=(const Video& v)
{
	invalidateCache();
	
	_path   = v._path;
	_frame  = v._frame;
	_labels = v._labels;
	
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
		
		image.setLabel(_getLabelForCurrentFrame());
		
        images.push_back(image);

		std::stringstream name;
		
		name << images.back().path() << "-frame-" << _frame;
	
		util::log("Video") << "Getting next frame " << _frame << " from video "
			<< path() << " with label '" << images.back().label() << "' \n";
	
		images.back().setPath(name.str());

		++_frame;
	}
	
	return images;
}

Image Video::getSpecificFrame(unsigned int frame)
{
	load();
	
	_seek(frame);
	
	Image image;
    
    if(_stream->getNextFrame(image))
	{
		image.setLabel(_getLabelForCurrentFrame());
	}

	util::log("Video") << "Getting specific frame " << frame << " from video "
		<< path() << " with label '" << image.label() << "' \n";
	
	return image;
}

size_t Video::getTotalFrames()
{
	load();

	return _stream->getTotalFrames();
}

bool Video::finished()
{
	load();
	
	return _stream->finished();
}

Video::LabelVector Video::getLabels() const
{
	return _labels;
}

void Video::addLabel(const Label& l)
{
	_labels.push_back(l);
}

const std::string& Video::path() const
{
	return _path;
}

bool Video::isPathAVideo(const std::string& path)
{
	auto extension = util::getExtension(path);
	
	return VideoLibraryInterface::isVideoTypeSupported(extension);
}

void Video::_seek(unsigned int frame)
{
	_stream->seek(frame);
	_frame = frame;
}

std::string Video::_getLabelForCurrentFrame() const
{
	std::string labelName = "unknown";
	
	for(auto& label : _labels)
	{
		if(label.beginFrame > _frame || label.endFrame < _frame) continue;
		
		return label.name;
	}
	
	return labelName;
}

Video::Label::Label(const std::string& n, unsigned int b, unsigned int e)
: name(n), beginFrame(b), endFrame(e)
{
	
}

unsigned int Video::Label::coveredFrames() const
{
	return endFrame - beginFrame;
}

}

}


