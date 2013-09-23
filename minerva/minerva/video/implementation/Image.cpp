/*	\file   Image.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Image class.
*/

// Minerva Includes
#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/ImageLibraryInterface.h>

#include <minerva/util/interface/paths.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cstring>
#include <cassert>
#include <cmath>

namespace minerva
{

namespace video
{

Image::Image(const std::string& path, const std::string& label)
: _path(path), _label(label), _loaded(false), _invalidToLoad(false), _x(0),
	_y(0), _colorComponents(0), _pixelSize(0)
{
	_loadImageHeader();
}

Image::Image(size_t x, size_t y, size_t c, size_t p, const std::string& path,
	const ByteVector& d, const std::string& l)
: _path(path), _label(l), _loaded(true), _invalidToLoad(true), _x(x), _y(y),
	_colorComponents(c), _pixelSize(p), _pixels(d)
{
	_pixels.resize(totalSize() * pixelSize());
}

size_t Image::x() const
{
	return _x;
}

size_t Image::y() const
{
	return _y;
}
	
size_t Image::colorComponents() const
{
	return _colorComponents;
}

size_t Image::pixelSize() const
{
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

void Image::displayOnScreen()
{
    load();
    ImageLibraryInterface::displayOnScreen(_x, _y, _colorComponents,
    	_pixelSize, _pixels);
}


bool Image::loaded() const
{
	return _loaded;
}

void Image::load()
{
	if(loaded()) return;
	
	util::log("Image") << "Loading data from image path '" << _path << "'\n";
	
	_pixels = ImageLibraryInterface::loadData(_path);

	util::log("Image") << " " << _pixels.size() << " bytes...\n";
	
	_loaded = true;
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

float Image::range() const
{
   return (float)(1 << (8 * pixelSize()));
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

Image::FloatVector Image::getSampledData(size_t sampleCount) const
{
	FloatVector samples;

	size_t imageSize = totalSize();
	double step = (imageSize + 0.0) / sampleCount;
	
	double samplePosition = 0.0;
	
	for(size_t sample = 0; sample != sampleCount;
		++sample, samplePosition += step)
	{
		size_t position = (size_t) samplePosition;
		
		if(position >= imageSize)
		{
			position = imageSize - 1;
		}
		
		samples.push_back((getComponentAt(position) * 2.0f / range()) - 1.0f);
	}

	return samples;
}

void Image::updateImageFromSamples(const FloatVector& samples)
{
	// nearest neighbor sampling
	for(size_t y = 0; y < this->y(); ++y)
	{
		for(size_t x = 0; x < this->x(); ++x)
		{
			for(size_t color = 0; color < this->colorComponents(); ++color)
			{
				size_t position = getPosition(x, y, color);
				size_t sampleIndex =
					((position + 0.0) / totalSize()) * samples.size();
				
				float sample = samples[sampleIndex];
				
				setComponentAt(x, y, color, sample);
			}
		}
	}
}

Image Image::sample(size_t samples) const
{
	// Nearest neighbor sampling
	
	double ratio = std::sqrt(samples / (totalSize() + 0.0));
	
	size_t newX = x() * ratio;
	size_t newY = y() * ratio;
	
	Image newImage(newX, newY, colorComponents(), pixelSize(), path());

	double xStep = (x() + 0.0) / newX;
	double yStep = (y() + 0.0) / newY;
	
	double yPosition = 0.0;
	
	for(size_t y = 0; y != newY; ++y, yPosition += yStep)
	{
		double xPosition = 0.0;

		for(size_t x = 0; x != newX; ++x, xPosition += xStep)
		{
			for(size_t color = 0; color != colorComponents(); ++color)
			{
				newImage.setComponentAt(x, y, color,
					getComponentAt(xPosition, yPosition, color));
			}
		}
	}
	
	return newImage;
}

float Image::getComponentAt(size_t position) const
{
	assert(loaded());

	float component = 0.0f;
	
	size_t positionInBytes = position * pixelSize();
	assertM(positionInBytes + pixelSize() <= _pixels.size(),
		"Position in bytes is " << positionInBytes << "");
	
	unsigned int value = 0;
	assert(pixelSize() < sizeof(unsigned int));
	
	std::memcpy(&value, &_pixels[positionInBytes], pixelSize());
	
	component = value;
	
	return component;
}

float Image::getComponentAt(size_t x, size_t y, size_t color) const
{
	return getComponentAt(getPosition(x, y, color));
}

void Image::setComponentAt(size_t x, size_t y, size_t color,
	float component)
{
	size_t positionInBytes = getPosition(x, y, color);
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


