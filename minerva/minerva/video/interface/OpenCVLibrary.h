/*	\file   OpenCVLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the OpenCVLibrary class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace video { class CvCapture; } }

namespace minerva
{

namespace video
{

class OpenCVLibrary
{
public:
	typedef struct _IplImage
	{
		int  nSize;
		int  ID;
		int  nChannels;
		int  alphaChannel;
		int  depth;
		char colorModel[4];
		char channelSeq[4];
		int  dataOrder;
		int  origin;
		int  align;
		int  width;
		int  height;
		struct _IplROI *roi;
		struct _IplImage *maskROI;
		void  *imageId;
		struct _IplTileInfo *tileInfo;
		int  imageSize;
		char *imageData;
		int  widthStep;
		int  BorderMode[4];
		int  BorderConst[4];
		char *imageDataOrigin;
	} IplImage;

	typedef IplImage Image;
    typedef void CvArr;
	
    static const int CV_LOAD_IMAGE_COLOR = 1;

	static const int IPL_DEPTH_SIGN = 0x80000000;
	static const int IPL_DEPTH_1U = 1;
	static const int IPL_DEPTH_8U = 8;
	static const int IPL_DEPTH_16U = 16;
	static const int IPL_DEPTH_32F = 32;
	static const int IPL_DEPTH_8S = (IPL_DEPTH_SIGN| 8);
	static const int IPL_DEPTH_16S = (IPL_DEPTH_SIGN|16);
	static const int IPL_DEPTH_32S = (IPL_DEPTH_SIGN|32);

	static const int IPL_DATA_ORDER_PIXEL = 0;


	static const int CV_CAP_PROP_POS_AVI_RATIO = 2;
    static const int CV_WINDOW_AUTOSIZE = 1;

public:
	static void load();
	static bool loaded();

public:
	static IplImage* cvLoadImage(const char* filename,
		int iscolor = CV_LOAD_IMAGE_COLOR);
	static int cvSaveImage(const char* filename, const IplImage* image);
	
	static void cvReleaseImage(IplImage**);

public:
	static CvCapture* cvCreateFileCapture(const char* filename);
	static void cvReleaseCapture(CvCapture**);
	
	static double cvGetCaptureProperty(CvCapture*, int property);
	static bool cvGrabFrame(CvCapture*);
	static IplImage* cvRetrieveFrame(CvCapture*);
    static int cvNamedWindow(const char* name, int flags=CV_WINDOW_AUTOSIZE);
    static void cvShowImage(const char* name, const IplImage* image);

private:
	static void _check();
	
private:
	class Interface
	{
	public:
		IplImage* (*cvLoadImage)(const char* filename, int iscolor);
		int (*cvSaveImage)(const char* filename, const IplImage*);
		void (*cvReleaseImage)(IplImage**);

	public:
		CvCapture* (*cvCreateFileCapture)(const char* filename);
		void (*cvReleaseCapture)(CvCapture**);
	
		double (*cvGetCaptureProperty)(CvCapture*, int property);
		bool (*cvGrabFrame)(CvCapture*);
		IplImage* (*cvRetrieveFrame)(CvCapture*);
        int (*cvNamedWindow)(const char* name, int flags);
        void (*cvShowImage) (const char* name, const IplImage* image);
    public:
		/*! \brief The constructor zeros out all of the pointers */
		Interface();
		
		/*! \brief The destructor closes dlls */
		~Interface();
		/*! \brief Load the library */
		void load();
		/*! \brief Has the library been loaded? */
		bool loaded() const;
		/*! \brief unloads the library */
		void unload();
				
	private:
		
		void* _library;
	};
	
private:
	static Interface _interface;

};

}

}


