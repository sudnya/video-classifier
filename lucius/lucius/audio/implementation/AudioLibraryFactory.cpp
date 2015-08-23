/*	\file   AudioLibraryFactory.cpp
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the AudioLibraryFactory class.
*/

// Lucius Includes
#include <lucius/audio/interface/AudioLibraryFactory.h>

#include <lucius/audio/interface/LibavcodecAudioLibrary.h>

namespace lucius
{

namespace audio
{

std::unique_ptr<AudioLibrary> AudioLibraryFactory::create(const std::string& name)
{
	if(name == "LibavcodecAudioLibrary")
	{
		return std::unique_ptr<AudioLibrary>(new LibavcodecAudioLibrary);
	}

	return std::unique_ptr<AudioLibrary>(nullptr);
}

AudioLibraryFactory::AudioLibraryVector AudioLibraryFactory::createAll()
{
	AudioLibraryVector libraries;

	libraries.emplace_back(create("LibavcodecAudioLibrary"));

	return libraries;
}

}

}



