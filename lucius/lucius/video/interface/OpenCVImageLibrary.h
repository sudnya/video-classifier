/*    \file   OpenCVImageLibrary.h
    \date   Thursday August 15, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the OpenCVImageLibrary class.
*/

#pragma once

// Lucius Includes
#include <lucius/video/interface/ImageLibrary.h>

namespace lucius
{

namespace video
{

class OpenCVImageLibrary : public ImageLibrary
{
public:
    virtual Header     loadHeader(const std::string& path);
    virtual DataVector loadData  (const std::string& path);

public:
    virtual void saveImage(const std::string& path, const Header& header,
        const DataVector& data);

public:
    virtual void displayOnScreen(size_t x, size_t y, size_t colorComponents,
        size_t pixelSize, const DataVector& pixels);
    virtual void deleteWindow();
    virtual void waitForKey(int delayInMilliseconds);
    virtual void addTextToStatusBar(const std::string& text);

public:
    virtual StringVector getSupportedExtensions() const;


};

}

}


