/*! \file SystemCompatibility.h
	\date Monday August 2, 2010
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for hacked code required to assist windows 
		compilaiton
*/

#pragma once

// Standard Library Includes
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace minerva
{

namespace util
{



/*! \brief Get the number of hardware threads */
unsigned int getHardwareThreadCount();
/*! \brief Get the full path to the named executable */
std::string getExecutablePath(const std::string& executableName);
/*! \brief The amount of free physical memory */
long long unsigned int getFreePhysicalMemory();
/*! \brief Get an estimate of the machine Floating Point Operations per second */
long long unsigned int getMachineFlops();
/*! \brief Has there been an OpenGL context bound to this process */
bool isAnOpenGLContextAvailable();
/*! \brief Is a string name mangled? */
bool isMangledCXXString(const std::string& string);
/*! \brief Demangle a string */
std::string demangleCXXString(const std::string& string);

}

}


