/*	\file   LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the
		LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary class.
*/

#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary.h>
#include <minerva/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace minerva
{

namespace optimizer
{

void LBFGSSolverLibrary::load()
{
	_interface.load();
}

bool LBFGSSolverLibrary::loaded()
{
	return _interface.loaded();
}

int LBFGSSolverLibrary::lbfgs(int n, double* x, double* ptr_fx,
	lbfgs_evaluate_t proc_evaluate, lbfgs_progress_t proc_progress,
	void* instance, lbfgs_parameter_t* param)
{
	_check();
	
	return (*_interface.lbfgs)(n, x, ptr_fx, proc_evaluate, proc_progress,
		instance, param);
}

void LBFGSSolverLibrary::lbfgs_parameter_init(lbfgs_parameter_t* param)
{
	_check();
	
	(*_interface.lbfgs_parameter_init)(param);
}

double* LBFGSSolverLibrary::lbfgs_malloc(int n)
{
	_check();
	
	return (*_interface.lbfgs_malloc)(n);
}

void LBFGSSolverLibrary::lbfgs_free(double* x)
{
	_check();

	(*_interface.lbfgs_free)(x);
}

void LBFGSSolverLibrary::_check()
{
	load();
	
	if(!loaded())
	{
		throw std::runtime_error("Tried to call opencv function when "
			"the library is not loaded. Loading library failed, consider "
			"installing liblbfgs.");
	}
}

LBFGSSolverLibrary::Interface::Interface()
: _library(nullptr)
{

}

LBFGSSolverLibrary::Interface::~Interface()
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

void LBFGSSolverLibrary::Interface::load()
{
	if(loaded()) return;
	
    #ifdef __APPLE__
    const char* libraryName = "liblbfgs.dylib";
    #else
    const char* libraryName = "liblbfgs.so";
    #endif

	_library = dlopen(libraryName, RTLD_LAZY);

    util::log("LBFGSSolverLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
	{
		return;
	}
	
	#define DynLink( function ) util::bit_cast(function, dlsym(_library, #function)); checkFunction((void*)function, #function)
	
	DynLink(lbfgs);
	
	DynLink(lbfgs_parameter_init);
	
	DynLink(lbfgs_malloc);
	DynLink(lbfgs_free);
	
	#undef DynLink	

	util::log("LBFGSSolverLibrary") << " success\n";
}

bool LBFGSSolverLibrary::Interface::loaded() const
{
	return _library != nullptr;
}

void LBFGSSolverLibrary::Interface::unload()
{
	if(!loaded()) return;

	dlclose(_library);
	_library = nullptr;
}

LBFGSSolverLibrary::Interface LBFGSSolverLibrary::_interface;

}

}


