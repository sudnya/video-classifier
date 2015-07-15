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

namespace lucius
{

namespace video
{

class ImageLibraryInterface
{
public:
	class Header
	{
	public:
		Header(unsigned int x = 0, unsigned int y = 0,
			unsigned int c = 0, unsigned int p = 0);
	
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

public:
	static void saveImage(const std::string& path, const Header& header,
		const DataVector& data);

public:
	static void displayOnScreen(size_t x, size_t y, size_t colorComponents,
		size_t pixelSize, const DataVector& pixels);
	static void deleteWindow();

	static void waitForKey(int delayInMilliseconds = 0);
	static void addTextToStatusBar(const std::string& text);

};

}

}


