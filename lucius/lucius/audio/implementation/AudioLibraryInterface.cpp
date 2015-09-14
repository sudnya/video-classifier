/*	\file   AudioLibraryInterface.cpp
	\date   Thursday August 15, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the AudioLibraryInterface class.
*/

// Lucius Includes
#include <lucius/audio/interface/AudioLibraryInterface.h>

#include <lucius/audio/interface/AudioLibrary.h>
#include <lucius/audio/interface/AudioLibraryFactory.h>

#include <lucius/util/interface/paths.h>

// Standard Library Includes
#include <map>

namespace lucius
{

namespace audio
{

class AudioLibraryDatabase
{
public:
	typedef std::map<std::string, AudioLibrary*> ExtensionToLibraryMap;

public:
	AudioLibraryDatabase()
	{
		_libraries = AudioLibraryFactory::createAll();

		for(auto& library : _libraries)
		{
			auto formats = library->getSupportedExtensions();

			for(auto format : formats)
			{
				libraries.emplace(format, library.get());
			}
		}
	}

public:
	ExtensionToLibraryMap libraries;

private:
	AudioLibraryFactory::AudioLibraryVector _libraries;

};

static AudioLibraryDatabase database;

AudioLibraryInterface::Header::Header()
: Header(0,0,0)
{

}

AudioLibraryInterface::Header::Header(size_t _samples,
    size_t _bytesPerSample, size_t _samplingRate)
: samples(_samples), bytesPerSample(_bytesPerSample), samplingRate(_samplingRate)
{

}

bool AudioLibraryInterface::Header::operator==(const Header& header) const
{
    return (samples == header.samples) &&
           (bytesPerSample == header.bytesPerSample) &&
           (samplingRate == header.samplingRate);
}

bool AudioLibraryInterface::isAudioTypeSupported(const std::string& extension)
{
    return database.libraries.count(extension) != 0;
}

AudioLibraryInterface::HeaderAndData AudioLibraryInterface::loadAudio(std::istream& stream,
    const std::string& extension)
{
	auto library = database.libraries.find(extension);

	if(library == database.libraries.end())
	{
		throw std::runtime_error("No audio library can support extension '" +
			extension + "'");
	}

	return library->second->loadAudio(stream, extension);
}

void AudioLibraryInterface::saveAudio(std::ostream& stream, const std::string& extension,
    const Header& header, const DataVector& data)
{
	auto library = database.libraries.find(extension);

	if(library == database.libraries.end())
	{
		throw std::runtime_error("No audio library can support extension '" +
			extension + "'");
	}

	library->second->saveAudio(stream, extension, header, data);
}

}

}




