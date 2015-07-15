/*	\file   VideoLibraryInterface.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the VideoLibraryInterface class.
*/

// Lucius Includes
#include <lucius/video/interface/VideoLibraryInterface.h>
#include <lucius/video/interface/VideoLibraryFactory.h>
#include <lucius/video/interface/VideoLibrary.h>

#include <lucius/util/interface/paths.h>

// Standard Library Includes
#include <map>

namespace lucius
{

namespace video
{

class VideoLibraryDatabase
{
public:
	typedef std::map<std::string, VideoLibrary*> ExtensionToLibraryMap;
	
public:
	VideoLibraryDatabase()
	{
		_libraries = VideoLibraryFactory::createAll();
		
		for(auto library : _libraries)
		{
			auto formats = library->getSupportedExtensions();
			
			for(auto format : formats)
			{
				libraries.insert(std::make_pair(format, library));
			}
		}
	}
	
	~VideoLibraryDatabase()
	{
		for(auto library : _libraries)
		{
			delete library;
		}
	}

public:
	ExtensionToLibraryMap libraries;

private:
	VideoLibraryFactory::VideoLibraryVector _libraries;

};

static VideoLibraryDatabase database;

bool VideoLibraryInterface::isVideoTypeSupported(const std::string& extension)
{
	return database.libraries.count(extension) != 0;	
}

VideoLibrary* VideoLibraryInterface::getLibraryThatSupports(
	const std::string& path)
{
	auto extension = util::getExtension(path);
	
	auto library = database.libraries.find(extension);
	
	if(library == database.libraries.end())
	{
		return nullptr;
	}
	
	return library->second;
}

}

}


