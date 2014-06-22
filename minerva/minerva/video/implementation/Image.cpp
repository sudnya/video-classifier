/*	\file   Image.cpp
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Image class.
*/

// Minerva Includes
#include <minerva/video/interface/Image.h>
#include <minerva/video/interface/ImageLibraryInterface.h>

#include <minerva/matrix/interface/Matrix.h>

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

float Image::range() const
{
   return (float)((1 << (8 * pixelSize())) - 1);
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

matrix::Matrix Image::convertToStandardizedMatrix(size_t sampleCount, size_t xTileSize, size_t yTileSize) const
{
	auto samples = getSampledData(sampleCount, xTileSize, yTileSize);
	
	return matrix::Matrix(1, sampleCount, samples);
}

Image::FloatVector Image::getSampledData(size_t sampleCount,
	size_t xTileSize, size_t yTileSize) const
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

		size_t zPosition = linearToZOrder(position, xTileSize, yTileSize);
		
		float sampleValue = standardize(getComponentAt(zPosition));

		//std::cout << path() << ": sample " << sampleValue << " (" << getComponentAt(position)
		//	<< ") = (position " << position << ")\n";
		
		samples.push_back(sampleValue);
	}

	return samples;
}

void Image::updateImageFromSamples(const FloatVector& samples,
	size_t xTileSize, size_t yTileSize)
{
	// nearest neighbor sampling
	for(size_t y = 0; y < this->y(); ++y)
	{
		for(size_t x = 0; x < this->x(); ++x)
		{
			for(size_t color = 0; color < this->colorComponents(); ++color)
			{
				size_t position  = getPosition(x, y, color);
				size_t zPosition = zToLinearOrder(position, xTileSize, yTileSize);

				size_t sampleIndex =
					((zPosition + 0.0) / totalSize()) * samples.size();
				
				float sample = samples[sampleIndex];
	
				setStandardizedComponentAt(x, y, color, sample);
			
				//std::cout << path() << ": (" << x << ", " << y << ", color " << color
				//	<< ", position " << position << ") = sample " << sample
				//	<< " (" << getComponentAt(x, y, color) << ") \n";
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

float Image::getComponentAt(size_t position) const
{
	assert(loaded());

	float component = 0.0f;
	
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

float Image::getComponentAt(size_t x, size_t y, size_t color) const
{
	return getComponentAt(getPosition(x, y, color));
}

float Image::getStandardizedComponentAt(size_t x, size_t y, size_t color) const
{
	return standardize(getComponentAt(x, y, color));
}

void Image::setComponentAt(size_t x, size_t y, size_t color,
	float component)
{
	size_t positionInBytes = getPosition(x, y, color) * pixelSize();
	assert(positionInBytes + pixelSize() <= _pixels.size());
	
	unsigned int value = component;
	assert(pixelSize() < sizeof(unsigned int));

	std::memcpy(&_pixels[positionInBytes], &value, pixelSize());
}

void Image::setStandardizedComponentAt(size_t x, size_t y, size_t color, float component)
{
	setComponentAt(x, y, color, destandardize(component));
}

size_t Image::getPosition(size_t x, size_t y, size_t color) const
{
	assert(x     < this->x());
	assert(y     < this->y());
	assert(color < colorComponents());

	return y * this->x() * colorComponents() +
		x * colorComponents() + color;
}

size_t Image::linearToZOrder(size_t linearPosition, size_t _xTileSize, size_t _yTileSize) const
{
	assert(linearPosition < totalSize());
	
	size_t pixelPosition = linearPosition / colorComponents();
	size_t color         = linearPosition % colorComponents();	

	size_t tileRowSize = _yTileSize * x();

	size_t yTileId        = pixelPosition / tileRowSize;
	size_t remainingInRow = pixelPosition % tileRowSize;

	size_t yTileSize = _yTileSize;
	
	if((yTileId + 1) * yTileSize > y())
	{
		yTileSize = y() - yTileId * yTileSize;
	}

	size_t tileSize = _xTileSize * yTileSize;

	size_t xTileId      = remainingInRow / (tileSize);
	size_t offsetInTile = remainingInRow % (tileSize);

	size_t xTileSize = _xTileSize;

	if((xTileId + 1) * xTileSize > x())
	{
		xTileSize = x() - xTileId * xTileSize;
	}

	size_t yOffsetInTile = offsetInTile / xTileSize;
	size_t xOffsetInTile = offsetInTile % xTileSize;

	size_t finalPosition = (yTileId * tileRowSize) + (xTileId * _xTileSize) + (yOffsetInTile * x()) + xOffsetInTile;

	size_t zPosition = (finalPosition * colorComponents() + color);

	assert(zPosition < totalSize());
	
//	std::cout << path() << ": mapping linear position "
//		<< linearPosition << " (" << (pixelPosition % x()) << " x, "
//		<< (pixelPosition / x()) << " y, " << color
//		<< " color) (" << xTileId << " xTileId, " << yTileId
//		<< " yTileId) (" << xOffsetInTile << " xOffsetInTile, "
//		<< yOffsetInTile << " yOffsetInTile) (" << xTileSize
//		<< " xTileSize, " << yTileSize << " yTileSize) ("
//		<< remainingInRow << " remaining in row) to z order " << zPosition << "\n";

	return zPosition;
}

size_t Image::zToLinearOrder(size_t zPosition, size_t _xTileSize, size_t _yTileSize) const
{
	assert(zPosition < totalSize());
	
	size_t pixelPosition = zPosition / colorComponents();
	size_t color         = zPosition % colorComponents();	

	size_t xPosition = pixelPosition % x();
	size_t yPosition = pixelPosition / x();
	
	size_t xTileId = xPosition / _xTileSize;
	size_t yTileId = yPosition / _yTileSize;

	size_t xTileOffset = xPosition % _xTileSize;
	size_t yTileOffset = yPosition % _yTileSize;
	
	size_t yTileSize = _yTileSize;
	
	if((yTileId + 1) * yTileSize > y())
	{
		yTileSize = y() - yTileId * yTileSize;
	}
	
	size_t xTileSize = _xTileSize;
	
	if((xTileId + 1) * xTileSize > x())
	{
		xTileSize = x() - xTileId * xTileSize;
	}
	
	size_t linearPosition = (yTileId * (_yTileSize * x())) +
		(yTileSize * _xTileSize * xTileId) +
		(yTileOffset * xTileSize) + xTileOffset;

	size_t linearPositionWithColor = linearPosition * colorComponents() + color;
	
//	std::cout << path() << ": mapping z order "
//		<< zPosition << " to linear position " << linearPositionWithColor
//		<< " reverse mapping would be "
//		<< linearToZOrder(linearPositionWithColor, _xTileSize, _yTileSize) << "\n";
	
	return linearPositionWithColor;
}

float Image::standardize(float component) const
{
	return (component / range()) * 2.0f - 1.0f;
}

float Image::destandardize(float component) const
{
	component = std::min(component,  1.0f);
	component = std::max(component, -1.0f);

	float colorValue = (component + 1.0f) * (range() / 2.0f);
	
	return colorValue;
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


