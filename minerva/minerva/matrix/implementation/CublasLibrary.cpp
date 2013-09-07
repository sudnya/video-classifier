/*	\file   CublasLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the CublasLibrary class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CublasLibrary.h>
#include <minerva/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace minerva
{

namespace matrix
{

void CublasLibrary::load()
{
	_interface.load();
}

bool CublasLibrary::loaded()
{
	return _interface.loaded();
}


void CublasLibrary::cublasSgemm(char transa, char transb, int m, int n, int k, 
	float alpha, const float *A, int lda, 
	const float *B, int ldb, float beta, float *C, 
	int ldc)
{
	_check();
	
	(*_interface.cublasSgemm)(transa, transb, m, n, k, alpha, A, lda, B, ldb,
		beta, C, ldc);
}

void* CublasLibrary::cudaMalloc(size_t bytes)
{
	_check();

	void* address = nullptr;

	(*_interface.cudaMalloc)(&address, bytes);
		
	return address;
}

void CublasLibrary::cudaFree(void* ptr)
{
	_check();

	(*_interface.cudaFree)(ptr);
}

void CublasLibrary::cudaMemcpy(void* dest, const void* src, size_t bytes)
{
	_check();

	(*_interface.cudaMemcpy)(dest, src, bytes);
}


void CublasLibrary::_check()
{
	load();
	
	if(!loaded())
	{
		throw std::runtime_error("Tried to call CUBLAS function when "
			"the library is not loaded. Loading library failed, consider "
			"installing CUBLAS.");
	}
}

CublasLibrary::Interface::Interface()
: _library(nullptr), _failed(false)
{

}

CublasLibrary::Interface::~Interface()
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

void CublasLibrary::Interface::load()
{
	if(_failed)  return;
	if(loaded()) return;
	
    #ifdef __APPLE__
    //const char* libraryName = "libcublas.dylib";
    const char* libraryName = "libcublas-optimized.dylib";
    #else
    const char* libraryName = "libcublas.so";
    #endif

	_library = dlopen(libraryName, RTLD_LAZY);

    util::log("CublasLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
	{
		util::log("CublasLibrary") << " Failed to load library '" << libraryName
			<< "'\n";
		_failed = true;
		return;
	}
	
	#define DynLink( function ) util::bit_cast(function, \
		dlsym(_library, #function)); checkFunction((void*)function, #function)
	
	DynLink(cublasSgemm);
	
	DynLink(cudaMalloc);
	DynLink(cudaFree);
	DynLink(cudaMemcpy);
	
	#undef DynLink	

	util::log("CublasLibrary") << " Loaded library '" << libraryName
		<< "' successfully\n";
}

bool CublasLibrary::Interface::loaded() const
{
	return _library != nullptr;
}

void CublasLibrary::Interface::unload()
{
	if(!loaded()) return;

	dlclose(_library);
	_library = nullptr;
}
	
CublasLibrary::Interface CublasLibrary::_interface;

}

}


