/*	\file   ImageLibraryInterface.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ImageLibraryInterface class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>
#include <cstdint>

namespace minerva
{

namespace video
{

class ImageLibraryInterface
{
public:
	class Header
	{
	public:
		unsigned int x;
		unsigned int y;
		unsigned int colorComponents;
		unsigned int pixelSize;
	};

public:
	typedef std::vector<uint8_t> DataVector;

public:
	static bool isImageTypeSupported(const std::string& extension);

public:
	static Header loadHeader(const std::string& path);
	static DataVector loadData(const std::string& path);

    static void displayOnScreen(size_t x, size_t y, size_t colorComponents, size_t pixelSize, const DataVector& pixels);

};

}

}


