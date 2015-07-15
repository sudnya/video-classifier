/*	\file   Video.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Video class.
*/

#pragma once

// Lucius Includes
#include <lucius/video/interface/ImageVector.h>

// Standard Library Includes
#include <string>
#include <vector>

namespace lucius
{

namespace video
{

class VideoLibrary;
class VideoStream;

class Video
{
public:
	class Label
	{
	public:
		Label(const std::string& name, unsigned int beginFrame,
			unsigned int endFrame);

	public:
		unsigned int coveredFrames() const;

	public:
		std::string  name;
		unsigned int beginFrame;
		unsigned int endFrame;
	
	};

	typedef std::vector<Label> LabelVector;

public:
	explicit Video(const std::string& path = "");
	Video(const std::string& path, const std::string& label,
		unsigned int beginFrame, unsigned int endFrame);
	~Video();

public:
	Video(const Video&);
	Video& operator=(const Video&);
	
public:
	bool loaded() const;
	void load();
	
	void invalidateCache();

public:
	ImageVector getNextFrames(unsigned int frames = 1);
	Image getSpecificFrame(unsigned int frame);

	size_t getTotalFrames();
	
	bool finished();

public:
	LabelVector getLabels() const;
	
	void addLabel(const Label& l);

public:
	const std::string& path() const;

public:
	static bool isPathAVideo(const std::string& path);

private:
	void _seek(unsigned int frame);

private:
	std::string _getLabelForCurrentFrame() const;

private:
	std::string   _path;
	unsigned int  _frame;

private:
	LabelVector _labels;

private:
	VideoLibrary* _library;
	VideoStream*  _stream;

};

typedef std::vector<Video> VideoVector;

}

}


