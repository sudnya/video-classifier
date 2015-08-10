/*	\file   AudioLibraryFactory.cpp
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the AudioLibraryFactory class.
*/

// Lucius Includes
#include <lucius/video/interface/AudioLibraryFactory.h>
#include <lucius/video/interface/LibavcodecAudioLibrary.h>

namespace lucius
{

namespace video
{

std::unique_ptr<AudioLibrary> LibraryFactory::create(const std::string& name)
{
	if(name == "LibavcodecAudioLibrary")
	{
		return std::unique_ptr<AudioLibrary>(new LibavcodecAudioLibrary);
	}

	return std::make_unique<AudioLibrary>(nullptr);
}

AudioLibraryFactory::AudioLibraryVector AudioLibraryFactory::createAll()
{
	AudioLibraryVector libraries;

	libraries.emplace_back(create("LibavcodecAudioLibrary"));

	return libraries;
}

}

}



