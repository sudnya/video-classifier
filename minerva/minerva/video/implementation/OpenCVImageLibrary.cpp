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

typedef OpenCVImageLibrary::Header Header;
typedef OpenCVImageLibrary::DataVector DataVector;
typedef OpenCVLibrary::IplImage IplImage;

Header OpenCVImageLibrary::loadHeader(const std::string& path)
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

static void createIplImage(IplImage& iplImage, const Header& header,
	const DataVector& data)
{
	iplImage.nSize = sizeof(IplImage);
	iplImage.ID = 0;
	iplImage.nChannels = header.colorComponents;
	iplImage.alphaChannel = 0;
	iplImage.depth = header.pixelSize * 8;
	
	iplImage.dataOrder = OpenCVLibrary::IPL_DATA_ORDER_PIXEL;
	iplImage.origin = 0;
	iplImage.align  = header.x;
	iplImage.width  = header.x;
	iplImage.height = header.y;
	
	iplImage.roi = nullptr;
	iplImage.maskROI = nullptr;
	iplImage.imageId = nullptr;
	iplImage.tileInfo = nullptr;
	
	iplImage.imageSize = data.size();
	iplImage.imageData = (char*)data.data();
	iplImage.widthStep = header.x;
	iplImage.imageDataOrigin = (char*)data.data();
	
}

void OpenCVImageLibrary::saveImage(const std::string& path,
	const Header& header, const DataVector& data)
{
	IplImage iplImage;
	
	createIplImage(iplImage, header, data);
}

void OpenCVImageLibrary::displayOnScreen(size_t x, size_t y,
	size_t colorComponents, size_t pixelSize, const DataVector& pixels)
{
    //call create window
    const char* name = "display-window";
    int error = OpenCVLibrary::cvNamedWindow(name/*, flags*/);
    if (error == 0)
    {
        //convert the imageData into CVArr
		IplImage iplImage;
		
		createIplImage(iplImage, Header(x, y, colorComponents, pixelSize),
			pixels);
        
        //call CV imageshow
        OpenCVLibrary::cvShowImage(name, &iplImage);
    }
    else
    {
		throw std::runtime_error("Failed to display window with OpenCV.");

    }
}

OpenCVImageLibrary::StringVector
	OpenCVImageLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".jpg|.png|render", "|"));
}

}

}


