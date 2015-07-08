/*	\file   VideoLibraryFactory.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the VideoLibraryFactory class.
*/

// Lucious Includes
#include <lucious/video/interface/VideoLibraryFactory.h>
#include <lucious/video/interface/OpenCVVideoLibrary.h>

namespace lucious
{

namespace video
{

VideoLibrary* VideoLibraryFactory::create(const std::string& name)
{
	if(name == "OpenCVVideoLibrary")
	{
		return new OpenCVVideoLibrary;
	}

	return nullptr;
}

VideoLibraryFactory::VideoLibraryVector VideoLibraryFactory::createAll()
{
	VideoLibraryVector libraries;
	
	libraries.push_back(create("OpenCVVideoLibrary"));

	return libraries;
}

}

}




