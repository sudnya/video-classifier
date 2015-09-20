/*    \file   WavAudioLibrary.h
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the WavAudioLibrary class.
*/

#pragma once

// Lucius Includes
#include <lucius/audio/interface/AudioLibraryInterface.h>
#include <lucius/audio/interface/AudioLibrary.h>

namespace lucius
{

namespace audio
{

class WavAudioLibrary : public AudioLibrary
{
public:
    typedef std::vector<std::string> StringVector;

public:
    virtual ~WavAudioLibrary();

public:
    virtual HeaderAndData loadAudio(std::istream& stream, const std::string& format);

public:
    virtual void saveAudio(std::ostream& stream, const std::string& format, const Header& header,
        const DataVector& data);

public:
    virtual StringVector getSupportedExtensions() const;


};

}

}






