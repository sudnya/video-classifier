/*	\file   ImageLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ImageLibrary class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/ImageLibraryInterface.h>

namespace minerva
{

namespace video
{

class ImageLibrary
{
public:
	typedef ImageLibraryInterface::Header     Header;
	typedef ImageLibraryInterface::DataVector DataVector;
	typedef std::vector<std::string>          StringVector;
	
public:
	virtual ~ImageLibrary();
	
public:
	virtual Header     loadHeader(const std::string& path) = 0;
	virtual DataVector loadData  (const std::string& path) = 0;

public:
	virtual StringVector getSupportedExtensions() const = 0;


};

}

}


