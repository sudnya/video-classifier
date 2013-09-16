/*	\file   Image.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Image class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <cstdint>

namespace minerva
{

namespace video
{

class Image
{
public:
	typedef std::vector<uint8_t> ByteVector;
	typedef std::vector<float>   FloatVector;

public:
	Image(const std::string& path, const std::string& label = "");
	Image(size_t x = 0, size_t y = 0, size_t colorComponents = 0,
		size_t pixelSize = 0, const std::string& path = "",
		const ByteVector& data = ByteVector(), const std::string& label = "");

public:
	size_t x() const;
	size_t y() const;
	
	size_t colorComponents() const;
	size_t pixelSize() const;
	size_t totalSize() const;

	const std::string& path()  const;
	const std::string& label() const;

public:
    float range() const;

public:
    void displayOnScreen();
    bool loaded() const;
	void load();

public:
	void setPath(const std::string& path);
	void setLabel(const std::string& label);
	void invalidateCache();
	
public:
	ByteVector& getRawData();

public:
	FloatVector getSampledData(size_t samples) const;
    Image sample(size_t samples) const;
    
public:
	float getComponentAt(size_t position) const;
	float getComponentAt(size_t x, size_t y, size_t color) const;

	void setComponentAt(size_t x, size_t y, size_t color, float component);

public:
	size_t getPosition(size_t x, size_t y, size_t color) const;

public:
	static bool isPathAnImage(const std::string& path);
	
private:
	void _loadImageHeader();
	
private:
	std::string _path;
	std::string _label;
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


