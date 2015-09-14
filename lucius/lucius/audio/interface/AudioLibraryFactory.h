/*	\file   AudioLibraryFactory.h
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the AudioLibraryFactory class.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <string>
#include <memory>

// Forward Declarations
namespace lucius { namespace audio { class AudioLibrary; } }

namespace lucius
{

namespace audio
{

class AudioLibraryFactory
{
public:
	typedef std::vector<std::unique_ptr<AudioLibrary>> AudioLibraryVector;

public:
	static std::unique_ptr<AudioLibrary> create(const std::string& name);
	static AudioLibraryVector createAll();


};

}

}



