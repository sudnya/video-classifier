/*	\file   ImageLibraryInterface.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ImageLibraryInterface class.
*/

// Minerva Includes
#include <minerva/video/interface/ImageLibraryInterface.h>

#include <minerva/video/interface/ImageLibrary.h>
#include <minerva/video/interface/ImageLibraryFactory.h>

#include <minerva/util/interface/paths.h>

// Standard Library Includes
#include <map>
#include <stdexcept>

namespace minerva
{

namespace video
{

class ImageLibraryDatabase
{
public:
	typedef std::map<std::string, ImageLibrary*> ExtensionToLibraryMap;
	
public:
	ImageLibraryDatabase()
	{
		_libraries = ImageLibraryFactory::createAll();
		
		for(auto library : _libraries)
		{
			auto formats = library->getSupportedExtensions();
			
			for(auto format : formats)
			{
				libraries.insert(std::make_pair(format, library));
			}
		}
	}
	
	~ImageLibraryDatabase()
	{
		for(auto library : _libraries)
		{
			delete library;
		}
	}

public:
	ExtensionToLibraryMap libraries;

private:
	ImageLibraryFactory::ImageLibraryVector _libraries;

};


static ImageLibraryDatabase database;

bool ImageLibraryInterface::isImageTypeSupported(const std::string& extension)
{
	return database.libraries.count(extension) != 0;	
}

ImageLibraryInterface::Header ImageLibraryInterface::loadHeader(
	const std::string& path)
{
	auto extension = util::getExtension(path);
	
	auto library = database.libraries.find(extension);
	
	if(library == database.libraries.end())
	{
		throw std::runtime_error("No image library can support extension '" +
			extension + "'");
	}
	
	return library->second->loadHeader(path);
}

ImageLibraryInterface::DataVector ImageLibraryInterface::loadData(
	const std::string& path)
{
	auto extension = util::getExtension(path);
	
	auto library = database.libraries.find(extension);
	
	if(library == database.libraries.end())
	{
		throw std::runtime_error("No image library can support extension '" +
			extension + "'");
	}
	
	return library->second->loadData(path);
}

}

}


