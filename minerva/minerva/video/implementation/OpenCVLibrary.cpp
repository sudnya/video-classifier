/*	\file   OpenCVLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the OpenCVLibrary class.
*/

// Minerva Includes
#include <minerva/video/interface/OpenCVLibrary.h>
#include <minerva/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace minerva
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
	_library = dlopen("libopencv_legacy.2.4.5.dylib", RTLD_LAZY);
    #else
	_library = dlopen("libopencv_legacy.so", RTLD_LAZY);
    #endif
	
    if(!loaded())
	{
		return;
	}
	
	#define DynLink( function ) util::bit_cast(function, dlsym(_library, #function)); checkFunction((void*)function, #function)
	
	DynLink(cvLoadImage);
	DynLink(cvReleaseImage);
	
	DynLink(cvCreateFileCapture);
	DynLink(cvReleaseCapture);
	DynLink(cvGetCaptureProperty);
	DynLink(cvGrabFrame);
	DynLink(cvRetrieveFrame);
	
	#undef DynLink	
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


