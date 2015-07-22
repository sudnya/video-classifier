/*	\file   OpenCVLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the OpenCVLibrary class.
*/

// Lucius Includes
#include <lucius/video/interface/OpenCVLibrary.h>
#include <lucius/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace lucius
{

namespace video
{

void OpenCVLibrary::load()
{
	_interface.load();
}

bool OpenCVLibrary::loaded()
{
	return _interface.loaded();
}

OpenCVLibrary::IplImage* OpenCVLibrary::cvLoadImage(const char* f, int c)
{
	_check();

	return (*_interface.cvLoadImage)(f, c);
}

int OpenCVLibrary::cvSaveImage(const char* f, const IplImage* i, const int* p)
{
	_check();

	return (*_interface.cvSaveImage)(f, i, p);
}

void OpenCVLibrary::cvReleaseImage(IplImage** i)
{
	_check();

	(*_interface.cvReleaseImage)(i);
}

CvCapture* OpenCVLibrary::cvCreateFileCapture(const char* f)
{
	_check();

	return (*_interface.cvCreateFileCapture)(f);
}

CvCapture* OpenCVLibrary::cvCaptureFromCAM(int device)
{
	_check();

	return (*_interface.cvCreateCameraCapture)(device);
}

void OpenCVLibrary::cvReleaseCapture(CvCapture** c)
{
	_check();

	(*_interface.cvReleaseCapture)(c);
}

double OpenCVLibrary::cvGetCaptureProperty(CvCapture* c, int p)
{
	_check();

	return (*_interface.cvGetCaptureProperty)(c, p);
}

int OpenCVLibrary::cvSetCaptureProperty(CvCapture* c, int p, double v)
{
	_check();

	return (*_interface.cvSetCaptureProperty)(c, p, v);
}

bool OpenCVLibrary::cvGrabFrame(CvCapture* c)
{
	_check();

	return (*_interface.cvGrabFrame)(c);
}

OpenCVLibrary::IplImage* OpenCVLibrary::cvRetrieveFrame(CvCapture* c)
{
	_check();

	return (*_interface.cvRetrieveFrame)(c);
}

int OpenCVLibrary::cvNamedWindow(const char* name, int flags)
{
    _check();

    return (*_interface.cvNamedWindow)(name, flags);
}

void OpenCVLibrary::cvDestroyWindow(const char* name)
{
    _check();

    return (*_interface.cvDestroyWindow)(name);
}

void OpenCVLibrary::cvShowImage(const char* name, const IplImage* image)
{
    _check();

    (*_interface.cvShowImage)(name, image);
}

void OpenCVLibrary::cvWaitKey(int millisecondDelay)
{
	_check();

	(*_interface.cvWaitKey)(millisecondDelay);
}

int OpenCVLibrary::cvCreateTrackbar(const char* name,
	const char* window, int* value, int count, CvTrackbarCallback on_change)
{
	_check();

	return (*_interface.cvCreateTrackbar)(name, window, value, count, on_change);
}

void OpenCVLibrary::cvDisplayOverlay(const char* window, const char* text, int delayInMilliseconds)
{
	_check();

	(*_interface.cvDisplayOverlay)(window, text, delayInMilliseconds);
}

void OpenCVLibrary::_check()
{
	load();

	if(!loaded())
	{
		throw std::runtime_error("Tried to call opencv function when "
			"the library is not loaded. Loading library failed, consider "
			"installing OpenCV.");
	}
}

OpenCVLibrary::Interface::Interface()
: _library(nullptr)
{

}

OpenCVLibrary::Interface::~Interface()
{
	unload();
}


static void checkFunction(void* pointer, const std::string& name)
{
	if(pointer == nullptr)
	{
		throw std::runtime_error("Failed to load function '" + name +
			"' from dynamic library.");
	}
}

void OpenCVLibrary::Interface::load()
{
	if(loaded()) return;

    #ifdef __APPLE__
    const char* libraryName = "libopencv_legacy.dylib";
    #else
    const char* libraryName = "libopencv_legacy.so";
    #endif

	_library = dlopen(libraryName, RTLD_LAZY);

    util::log("OpenCVLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
	{
        util::log("OpenCVLibrary") << " Loading library '" << libraryName << "' failed\n";
		return;
	}

	#define DynLink( function ) util::bit_cast(function, dlsym(_library, #function)); checkFunction((void*)function, #function)

	DynLink(cvLoadImage);
	DynLink(cvSaveImage);
	DynLink(cvReleaseImage);

	DynLink(cvCreateFileCapture);
	DynLink(cvCreateCameraCapture);
	DynLink(cvReleaseCapture);
	DynLink(cvGetCaptureProperty);
	DynLink(cvSetCaptureProperty);
	DynLink(cvGrabFrame);
	DynLink(cvRetrieveFrame);

	DynLink(cvNamedWindow);
	DynLink(cvDestroyWindow);
	DynLink(cvShowImage);
	DynLink(cvWaitKey);
	DynLink(cvCreateTrackbar);

	// This is buggy on OSX
	//DynLink(cvDisplayOverlay);

	#undef DynLink

    util::log("OpenCVLibrary") << " Loaded library '" << libraryName << "' succeeded\n";
}

bool OpenCVLibrary::Interface::loaded() const
{
	return _library != nullptr;
}

void OpenCVLibrary::Interface::unload()
{
	if(!loaded()) return;

	dlclose(_library);
	_library = nullptr;
}

OpenCVLibrary::Interface OpenCVLibrary::_interface;

}

}


