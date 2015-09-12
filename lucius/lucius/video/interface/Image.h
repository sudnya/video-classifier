/*    \file   Image.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Image class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <cstdint>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace video
{

class Image
{
public:
    typedef std::vector<uint8_t> ByteVector;
    typedef std::vector<double>  FloatVector;
    typedef matrix::Matrix       Matrix;

public:
    Image(const std::string& path, const std::string& label = "");
    Image(size_t x = 0, size_t y = 0, size_t colorComponents = 0,
        size_t pixelSize = 0, const std::string& path = "",
        const std::string& label = "",
        const ByteVector& data = ByteVector());

public:
    void setTile(size_t x, size_t y, const Image& image);
    Image getTile(size_t x, size_t y, size_t xPixels, size_t yPixels, size_t colors) const;

public:
    size_t x() const;
    size_t y() const;

    size_t colorComponents() const;
    size_t pixelSize() const;
    size_t totalSize() const;

    const std::string& path()  const;
    const std::string& label() const;

public:
    bool hasLabel() const;

public:
    void displayOnScreen();
    void displayOnScreen() const;
    void addTextToDisplay(const std::string& text) const;
    void waitForKeyPress() const;
    void deleteWindow() const;

public:
    bool loaded() const;
    bool headerLoaded() const;
    void load();
    void loadHeader();
    void save();

public:
    void setPath(const std::string& path);
    void setLabel(const std::string& label);

public:
    void invalidateCache();

public:
    ByteVector& getRawData();

public:
    Image downsample(size_t x, size_t y, size_t colors) const;

public:
    double getComponentAt(size_t position) const;
    double getComponentAt(size_t x, size_t y, size_t color) const;

    void setComponentAt(size_t x, size_t y, size_t color, double component);
    void setComponentAt(size_t position, double component) const;

public:
    size_t getPosition(size_t x, size_t y, size_t color) const;

public:
    static bool isPathAnImage(const std::string& path);

private:
    void _loadImageHeader();

private:
    std::string _path;
    std::string _label;
    bool        _headerLoaded;
    bool        _loaded;
    bool        _invalidToLoad;

private:
    size_t _x;
    size_t _y;
    size_t _colorComponents;
    size_t _pixelSize;

    ByteVector _pixels;

};

}

}


