/*    \file   VideoLibraryInterface.h
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the VideoLibraryInterface class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucius { namespace video { class VideoLibrary; } }

namespace lucius
{

namespace video
{

class VideoLibraryInterface
{
public:
    static bool isVideoTypeSupported(const std::string& extension);

public:
    static VideoLibrary* getLibraryThatSupports(const std::string& path);
    static VideoLibrary* getLibraryThatSupportsCamera();

};

}

}


