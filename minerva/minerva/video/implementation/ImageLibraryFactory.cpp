/*	\file   ImageLibraryFactory.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ImageLibraryFactory class.
*/

// Minerva Includes
#include <minerva/video/interface/ImageLibraryFactory.h>
#include <minerva/video/interface/OpenCVImageLibrary.h>

namespace minerva
{

namespace video
{

ImageLibrary* ImageLibraryFactory::create(const std::string& name)
{
	if(name == "OpenCVImageLibrary")
	{
		return new OpenCVImageLibrary;
	}

	return nullptr;
}

ImageLibraryFactory::ImageLibraryVector ImageLibraryFactory::createAll()
{
	ImageLibraryVector libraries;
	
	libraries.push_back(create("OpenCVImageLibrary"));

	return libraries;
}

}

}


