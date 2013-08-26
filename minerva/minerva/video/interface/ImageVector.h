/*	\file   ImageVector.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Vector class.
*/

#pragma once

// Minerva Includes
#include <minerva/video/interface/Image.h>

#include <minerva/matrix/interface/Matrix.h>

// Standard Library Includes
#include <vector>

namespace minerva
{

namespace video
{

class ImageVector
{
public:
	typedef std::vector<Image>              BaseImageVector;
	typedef BaseImageVector::iterator       iterator;
	typedef BaseImageVector::const_iterator const_iterator;

	typedef matrix::Matrix Matrix;

public:
	ImageVector();
	~ImageVector();

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	      Image& operator[](size_t index);
	const Image& operator[](size_t index) const;

public:
	      Image& back();
	const Image& back() const;

public:
	size_t size() const;

public:
	void push_back(const Image& image);

public:
	Matrix convertToMatrix(size_t sampleCount) const;
    Matrix getReference() const;

private:
	size_t _getSampledImageSize() const;

private:
	BaseImageVector _images;

};

}

}


