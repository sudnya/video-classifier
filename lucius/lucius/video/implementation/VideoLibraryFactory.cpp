/*    \file   VideoLibraryFactory.cpp
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the VideoLibraryFactory class.
*/

// Lucius Includes
#include <lucius/video/interface/VideoLibraryFactory.h>
#include <lucius/video/interface/OpenCVVideoLibrary.h>

namespace lucius
{

namespace video
{

VideoLibrary* VideoLibraryFactory::create(const std::string& name)
{
    if(name == "OpenCVVideoLibrary")
    {
        return new OpenCVVideoLibrary;
    }

    return nullptr;
}

VideoLibraryFactory::VideoLibraryVector VideoLibraryFactory::createAll()
{
    VideoLibraryVector libraries;
    
    libraries.push_back(create("OpenCVVideoLibrary"));

    return libraries;
}

}

}




