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
	virtual void saveImage(const std::string& path, const Header& header,
		const DataVector& data) = 0;

public:
	virtual void displayOnScreen(size_t x, size_t y, size_t colorComponents,
		size_t pixelSize, const DataVector& pixels) = 0;
	virtual void deleteWindow() = 0;
	virtual void waitForKey(int delayInMilliseconds) = 0;
	virtual void addTextToStatusBar(const std::string& text) = 0;

public:
	virtual StringVector getSupportedExtensions() const = 0;


};

}

}


