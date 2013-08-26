/*	\file   Video.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Video class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/ImageVector.h>

// Standard Library Includes
#include <string>
#include <vector>

namespace minerva
{

namespace video
{

class VideoLibrary;
class VideoStream;

class Video
{
public:
	Video(const std::string& path = "");
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

	bool finished();

public:
	static bool isPathAVideo(const std::string& path);

private:
	void _seek(unsigned int frame);

private:
	std::string   _path;
	unsigned int  _frame;

private:
	VideoLibrary* _library;
	VideoStream*  _stream;

};

typedef std::vector<Video> VideoVector;

}

}


