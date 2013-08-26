/*	\file   ImageLibraryFactory.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ImageLibraryFactory class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>

// Forward Declarations
namespace minerva { namespace video { class ImageLibrary; } }

namespace minerva
{

namespace video
{

class ImageLibraryFactory
{
public:
	typedef std::vector<ImageLibrary*> ImageLibraryVector;
	
public:
	static ImageLibrary* create(const std::string& name);
	static ImageLibraryVector createAll();
	

};

}

}


