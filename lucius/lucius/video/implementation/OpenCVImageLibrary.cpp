/*	\file   OpenCVImageLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the OpenCVImageLibrary class.
*/

// Lucius Includes
#include <lucius/video/interface/OpenCVImageLibrary.h>
#include <lucius/video/interface/OpenCVLibrary.h>

#include <lucius/util/interface/string.h>

// Standard Library Interface
#include <stdexcept>

namespace lucius
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

	iplImage.colorModel[0] = 'R';
	iplImage.colorModel[1] = 'G';
	iplImage.colorModel[2] = 'B';
	iplImage.colorModel[3] = '\0';
	iplImage.channelSeq[0] = 'B';
	iplImage.channelSeq[1] = 'G';
	iplImage.channelSeq[2] = 'R';
	iplImage.channelSeq[3] = '\0';
	
	iplImage.dataOrder = OpenCVLibrary::IPL_DATA_ORDER_PIXEL;
	iplImage.origin = 0;
	iplImage.align  = 4;
	iplImage.width  = header.x;
	iplImage.height = header.y;
	

	iplImage.roi = nullptr;
	iplImage.maskROI = nullptr;
	iplImage.imageId = nullptr;
	iplImage.tileInfo = nullptr;

	iplImage.BorderMode[0] = 0;
	iplImage.BorderMode[1] = 0;
	iplImage.BorderMode[2] = 0;
	iplImage.BorderMode[3] = 0;
	iplImage.BorderConst[0] = 0;
	iplImage.BorderConst[1] = 0;
	iplImage.BorderConst[2] = 0;
	iplImage.BorderConst[3] = 0;
	
	iplImage.imageSize = data.size();
	iplImage.imageData = (char*)data.data();
	iplImage.widthStep = header.x * header.colorComponents;
	iplImage.imageDataOrigin = (char*)data.data();
	
}

void OpenCVImageLibrary::saveImage(const std::string& path,
	const Header& header, const DataVector& data)
{
	IplImage iplImage;
	
	createIplImage(iplImage, header, data);
	
	int status = OpenCVLibrary::cvSaveImage(path.c_str(), &iplImage);

	if(!status)
	{
		throw std::runtime_error("Failed to save image '" + path
			+ "' with OpenCV.");
	}
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

void OpenCVImageLibrary::deleteWindow()
{
    const char* name = "display-window";
	
	OpenCVLibrary::cvDestroyWindow(name);
}

void OpenCVImageLibrary::waitForKey(int delayInMilliseconds)
{
	OpenCVLibrary::cvWaitKey(delayInMilliseconds);
}

void OpenCVImageLibrary::addTextToStatusBar(const std::string& text)
{
	const char* name  = "display-window";

	// This is buggy and requires Qt
	// const int   delay = 0;
	// OpenCVLibrary::cvDisplayOverlay(name, text.c_str(), delay);

	OpenCVLibrary::cvCreateTrackbar(text.c_str(), name);
}

OpenCVImageLibrary::StringVector
	OpenCVImageLibrary::getSupportedExtensions() const
{
    return StringVector(util::split(".jpg|.png|render", "|"));
}

}

}


