/*    \file   AudioLibrary.h
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the AudioLibrary class.
*/

#pragma once

// Lucius Includes
#include <lucius/audio/interface/AudioLibraryInterface.h>

// Standard Library Includes
#include <istream>
#include <ostream>

namespace lucius
{

namespace audio
{

class AudioLibrary
{
public:
    typedef AudioLibraryInterface::Header        Header;
    typedef AudioLibraryInterface::HeaderAndData HeaderAndData;
    typedef AudioLibraryInterface::DataVector    DataVector;
    typedef std::vector<std::string>             StringVector;

public:
    virtual ~AudioLibrary();

public:
    virtual HeaderAndData loadAudio(std::istream& , const std::string& format) = 0;

public:
    virtual void saveAudio(std::ostream& path, const std::string& format, const Header& header,
        const DataVector& data) = 0;

public:
    virtual StringVector getSupportedExtensions() const = 0;


};

}

}




