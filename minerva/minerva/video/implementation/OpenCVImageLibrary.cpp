/*	\file   OpenCVImageLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the OpenCVImageLibrary class.
*/

// Minerva Includes
#include <minerva/video/interface/OpenCVImageLibrary.h>
#include <minerva/video/interface/OpenCVLibrary.h>

#include <minerva/util/interface/string.h>

// Standard Library Interface
#include <stdexcept>

namespace minerva
{

namespace video
{

OpenCVImageLibrary::Header OpenCVImageLibrary::loadHeader(
	const std::string& path)
{
	Header header;
	
	OpenCVLibrary::Image* image = OpenCVLibrary::cvLoadImage(path.c_str());
	
	if(image == nullptr)
	{
		throw std::runtime_error("Failed to open path '" + path +
			"' with OpenCV.");
	}
	
	header.x = image->width;
	header.y = image->height;
	header.colorComponents = image->nChannels;
	header.pixelSize = (image->depth % 32) / 8;
	
	OpenCVLibrary::cvReleaseImage(&image);
	
	return header;
}

OpenCVImageLibrary::DataVector OpenCVImageLibrary::loadData(
	const std::string& path)
{
	DataVector data;
	
	OpenCVLibrary::Image* image = OpenCVLibrary::cvLoadImage(path.c_str());
	
	if(image == nullptr)
	{
		throw std::runtime_error("Failed to open path '" + path +
			"' with OpenCV.");
	}
	
	int pixelSize = (image->depth % 32) / 8;
	
	for(int y = 0; y < image->height; ++y)
	{
		for(int x = 0; x < image->width; ++x)
		{
			for(int c = 0; c < image->nChannels; ++c)
			{
				for(int i = 0; i < pixelSize; ++i)
				{
					int position = 
						y * image->widthStep * pixelSize +
						x * pixelSize * image->nChannels + c * pixelSize + i;
				
					data.push_back(image->imageData[position]);
				}
			}
		}
	}
	
	OpenCVLibrary::cvReleaseImage(&image);
	
	return data;
}

OpenCVImageLibrary::StringVector
	OpenCVImageLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".jpg|.png", "|"));
}

}

}


