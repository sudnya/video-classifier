/*	\file   OpenCVImageLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the OpenCVImageLibrary class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/ImageLibrary.h>

namespace minerva
{

namespace video
{

class OpenCVImageLibrary : public ImageLibrary
{
public:
	virtual Header     loadHeader(const std::string& path);
	virtual DataVector loadData  (const std::string& path);

public:
	virtual StringVector getSupportedExtensions() const;
    virtual void displayOnScreen(size_t x, size_t y, size_t colorComponents, size_t pixelSize, const DataVector& pixels);


};

}

}


