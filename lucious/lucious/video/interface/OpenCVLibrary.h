/*	\file   OpenCVLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the OpenCVLibrary class.
*/

#pragma once

// Forward Declarations
namespace lucious { namespace video { class CvCapture; } }

namespace lucious
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
	typedef void* CvTrackbarCallback;
	
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
	static const int CV_CAP_PROP_FRAME_COUNT   = 7;
	static const int CV_CAP_PROP_POS_FRAMES    = 1;
	
    static const int CV_WINDOW_AUTOSIZE = 1;

public:
	static void load();
	static bool loaded();

public:
	static IplImage* cvLoadImage(const char* filename,
		int iscolor = CV_LOAD_IMAGE_COLOR);
	static int cvSaveImage(const char* filename, const IplImage* image, const int* params = nullptr);
	
	static void cvReleaseImage(IplImage**);

public:
	static CvCapture* cvCreateFileCapture(const char* filename);
	static void cvReleaseCapture(CvCapture**);
	
	static double cvGetCaptureProperty(CvCapture*, int property);
	static int cvSetCaptureProperty(CvCapture*, int property, double value);
	static bool cvGrabFrame(CvCapture*);
	static IplImage* cvRetrieveFrame(CvCapture*);

public:
    static int cvNamedWindow(const char* name, int flags=CV_WINDOW_AUTOSIZE);
    static void cvDestroyWindow(const char* name);
    static void cvShowImage(const char* name, const IplImage* image);
    static void cvWaitKey(int millisecondDelay = 0);
	static int cvCreateTrackbar(const char* trackbar_name,
		const char* window_name, int* value = nullptr, int count = 100, CvTrackbarCallback on_change = nullptr);	

public:
	static void cvDisplayOverlay(const char* name, const char* text, int delayInMilliseconds);   
	

private:
	static void _check();
	
private:
	class Interface
	{
	public:
		IplImage* (*cvLoadImage)(const char* filename, int iscolor);
		int (*cvSaveImage)(const char* filename, const IplImage*, const int* params);
		void (*cvReleaseImage)(IplImage**);

	public:
		CvCapture* (*cvCreateFileCapture)(const char* filename);
		void (*cvReleaseCapture)(CvCapture**);
	
		double (*cvGetCaptureProperty)(CvCapture*, int property);
		int (*cvSetCaptureProperty)(CvCapture*, int property, double value);
		bool (*cvGrabFrame)(CvCapture*);
		IplImage* (*cvRetrieveFrame)(CvCapture*);
        int (*cvNamedWindow)(const char* name, int flags);
        void (*cvDestroyWindow)(const char* name);
		void (*cvShowImage) (const char* name, const IplImage* image);
    	void (*cvWaitKey) (int millisecondDelay);
		int (*cvCreateTrackbar) (const char* trackbar_name,
			const char* window_name, int* value, int count, CvTrackbarCallback on_change);	
		void (*cvDisplayOverlay) (const char* name, const char* text, int delayInMilliseconds);   
		
 
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


