/*	\file   AudioLibraryInterface.cpp
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the AudioLibraryInterface class.
*/

// Lucius Includes
#include <lucius/audio/interface/AudioLibraryInterface.h>

#include <lucius/audio/interface/AudioLibrary.h>
#include <lucius/audio/interface/AudioLibraryFactory.h>

namespace lucius
{

namespace audio
{

class AudioLibraryDatabase
{
public:
	typedef std::map<std::string, std::unique_ptr<ImageLibrary>> ExtensionToLibraryMap;

public:
	AudioLibraryDatabase()
	{
		_libraries = AudioLibraryFactory::createAll();

		for(auto& library : _libraries)
		{
			auto formats = library->getSupportedExtensions();

			for(auto format : formats)
			{
				libraries.emplace(format, std::move(library));
			}
		}
	}

public:
	ExtensionToLibraryMap libraries;

private:
	AudioLibraryFactory::AudioLibraryVector _libraries;

};

static AudioLibraryDatabase database;

AudioLibraryInterface::Header::Header(size_t _samples,
    size_t _bytesPerSample, size_t _samplingRate);
: samples(_samples), bytesPerSample(_bytesPerSample), samplingRate(_samplingRate)
{

}

bool AudioLibraryInterface::isAudioTypeSupported(const std::string& extension)
{
    return database.libraries.count(extension) != 0;
}

AudioLibraryInterface::Header AudioLibraryInterface::loadHeader(const std::string& path)
{
	auto extension = util::getExtension(path);

	auto library = database.libraries.find(extension);

	if(library == database.libraries.end())
	{
		throw std::runtime_error("No audio library can support extension '" +
			extension + "'");
	}

	return library->second->loadHeader(path);
}

AudioLibraryInterface::DataVector AudioLibraryInterface::loadData(const std::string& path)
{
	auto extension = util::getExtension(path);

	auto library = database.libraries.find(extension);

	if(library == database.libraries.end())
	{
		throw std::runtime_error("No audio library can support extension '" +
			extension + "'");
	}

	return library->second->loadData(path);
}

void AudioLibraryInterface::saveAudio(const std::string& path, const Header& header,
    const DataVector& data)
{
	auto extension = util::getExtension(path);

	auto library = database.libraries.find(extension);

	if(library == database.libraries.end())
	{
		throw std::runtime_error("No audio library can support extension '" +
			extension + "'");
	}

	library->second->saveAudio(path, header, data);
}

}

}




