/*	\file   Image.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Image class.
*/

// Lucious Includes
#include <lucious/video/interface/Image.h>
#include <lucious/video/interface/ImageLibraryInterface.h>

#include <lucious/matrix/interface/Matrix.h>

#include <lucious/util/interface/paths.h>
#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <cstring>
#include <cassert>
#include <cmath>

namespace lucious
{

namespace video
{

Image::Image(const std::string& path, const std::string& label)
: _path(path), _label(label), _headerLoaded(false), _loaded(false),
	_invalidToLoad(false), _x(0), _y(0), _colorComponents(0), _pixelSize(0)
{

}

Image::Image(size_t x, size_t y, size_t c, size_t p, const std::string& path,
	const std::string& l, const ByteVector& d)
: _path(path), _label(l), _headerLoaded(true), _loaded(true), _invalidToLoad(true),
	_x(x), _y(y), _colorComponents(c), _pixelSize(p), _pixels(d)
{
	_pixels.resize(totalSize() * pixelSize());
}

void Image::setTile(size_t xStart, size_t yStart, const Image& image)
{
	// TODO faster
	size_t xSize     = image.x();
	size_t ySize     = image.y();
	size_t colorSize = image.colorComponents();

	assert(xStart + xSize <= x());
	assert(yStart + ySize <= y());
	assert(colorSize <= colorComponents());

	for(size_t y = 0; y < ySize; ++y)
	{
		for(size_t x = 0; x < xSize; ++x)
		{
			for(size_t c = 0; c < colorSize; ++c)
			{
				setComponentAt(xStart + x, yStart + y, c, image.getComponentAt(x, y, c));
			}
		}
	}
}

Image Image::getTile(size_t xStart, size_t yStart, size_t xPixels, size_t yPixels, size_t colors) const
{
	Image result(xPixels, yPixels, colors, pixelSize(), path(), label());

	// TODO faster

	for(size_t y = 0; y < yPixels; ++y)
	{
		for(size_t x = 0; x < xPixels; ++x)
		{
			for(size_t c = 0; c < colors; ++c)
			{
				size_t color = c % colorComponents();

				result.setComponentAt(x, y, c, getComponentAt(x + xStart, y + yStart, color));
			}
		}
	}

	return result;
}

size_t Image::x() const
{
	assert(_headerLoaded);

	return _x;
}

size_t Image::y() const
{
	assert(_headerLoaded);

	return _y;
}

size_t Image::colorComponents() const
{
	assert(_headerLoaded);

	return _colorComponents;
}

size_t Image::pixelSize() const
{
	assert(_headerLoaded);

	return _pixelSize;
}

size_t Image::totalSize() const
{
	return x() * y() * colorComponents();
}

const std::string& Image::path() const
{
	return _path;
}

const std::string& Image::label() const
{
	return _label;
}

bool Image::hasLabel() const
{
	return !label().empty();
}

void Image::displayOnScreen()
{
    load();
    ImageLibraryInterface::displayOnScreen(x(), y(), colorComponents(),
    	pixelSize(), _pixels);
}

void Image::displayOnScreen() const
{
    assert(loaded());

    ImageLibraryInterface::displayOnScreen(x(), y(), colorComponents(),
    	pixelSize(), _pixels);
}

void Image::waitForKeyPress() const
{
	ImageLibraryInterface::waitForKey();
}

void Image::addTextToDisplay(const std::string& text) const
{
	ImageLibraryInterface::addTextToStatusBar(text);
}

void Image::deleteWindow() const
{
	ImageLibraryInterface::deleteWindow();
}

bool Image::loaded() const
{
	return _loaded;
}

bool Image::headerLoaded() const
{
	return _headerLoaded;
}

void Image::load()
{
	if(loaded()) return;

	loadHeader();

	util::log("Image") << "Loading data from image path '" << _path << "'\n";

	_pixels = ImageLibraryInterface::loadData(_path);

	util::log("Image") << " " << _pixels.size() << " bytes...\n";

	_loaded = true;
}

void Image::loadHeader()
{
	if(_headerLoaded) return;

	_loadImageHeader();

	_headerLoaded = true;
}

void Image::save()
{
	if(!loaded()) return;

	util::log("Image") << "Saving data to image path '" << _path << "'\n";

	ImageLibraryInterface::saveImage(_path,
		ImageLibraryInterface::Header(x(), y(), colorComponents(), pixelSize()),
		_pixels);

	util::log("Image") << " " << _pixels.size() << " bytes...\n";

	_loaded = true;
}

void Image::setPath(const std::string& path)
{
	if(!_invalidToLoad)
	{
		invalidateCache();
	}

	_path = path;
}

void Image::setLabel(const std::string& label)
{
	_label = label;
}

void Image::invalidateCache()
{
	if(!loaded()) return;

	assert(!_invalidToLoad);

	// free the storage
	ByteVector().swap(_pixels);

	_loaded = false;
}

Image::ByteVector& Image::getRawData()
{
	return _pixels;
}

Image Image::downsample(size_t newX, size_t newY, size_t newColorComponents) const
{
	Image image(newX, newY, newColorComponents, pixelSize(), path(), label());

	double xStep     = (x() + 0.0) / newX;
	double yStep     = (y() + 0.0) / newY;
	double colorStep = (colorComponents() + 0.0) / newColorComponents;

	double yPosition = 0.0;

	for(size_t y = 0; y != newY; ++y, yPosition += yStep)
	{
		double xPosition = 0.0;

		for(size_t x = 0; x != newX; ++x, xPosition += xStep)
		{
			double colorPosition = 0.0;

			for(size_t color = 0; color != newColorComponents;
				++color, colorPosition += colorStep)
			{
				image.setComponentAt(x, y, color,
					getComponentAt(xPosition, yPosition, colorPosition));
			}
		}
	}

	return image;
}

double Image::getComponentAt(size_t position) const
{
	assert(loaded());

	double component = 0.0;

	size_t positionInBytes = position * pixelSize();
	assertM(positionInBytes + pixelSize() <= _pixels.size(),
		"Position in bytes is " << positionInBytes
		<< " out of max size " << _pixels.size());

	unsigned int value = 0;
	assert(pixelSize() < sizeof(unsigned int));

	std::memcpy(&value, &_pixels[positionInBytes], pixelSize());

	component = value;

	return component;
}

double Image::getComponentAt(size_t x, size_t y, size_t color) const
{
	return getComponentAt(getPosition(x, y, color));
}

void Image::setComponentAt(size_t x, size_t y, size_t color,
	double component)
{
	size_t positionInBytes = getPosition(x, y, color) * pixelSize();
	assert(positionInBytes + pixelSize() <= _pixels.size());

	unsigned int value = component;
	assert(pixelSize() < sizeof(unsigned int));

	std::memcpy(&_pixels[positionInBytes], &value, pixelSize());
}

size_t Image::getPosition(size_t x, size_t y, size_t color) const
{
	assert(x     < this->x());
	assert(y     < this->y());
	assert(color < colorComponents());

	return y * this->x() * colorComponents() +
		x * colorComponents() + color;
}

bool Image::isPathAnImage(const std::string& path)
{
	auto extension = util::getExtension(path);

	return ImageLibraryInterface::isImageTypeSupported(extension);
}

void Image::_loadImageHeader()
{
	util::log("Image") << "Loading header from image path '" << _path << "'\n";

	auto header = ImageLibraryInterface::loadHeader(_path);

	_x = header.x;
	_y = header.y;

	_colorComponents = header.colorComponents;

	_pixelSize = header.pixelSize;

	util::log("Image") << " (" << _x <<  " x pixels, " << _y <<  " y pixels, "
		<< _colorComponents <<  " components per pixel, " << _pixelSize
		<<  " bytes per pixel)\n";


}

}

}


