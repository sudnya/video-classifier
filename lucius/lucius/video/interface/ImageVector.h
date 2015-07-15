/*	\file   ImageVector.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Vector class.
*/

#pragma once

// Lucius Includes
#include <lucius/video/interface/Image.h>

#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/string.h>

// Standard Library Includes
#include <vector>
#include <random>

// Forward Declarations
namespace lucius { namespace neuralnetwork { class NeuralNetwork; } }

namespace lucius
{

namespace video
{

class ImageVector
{
public:
	typedef std::vector<Image>              BaseImageVector;
	typedef BaseImageVector::iterator       iterator;
	typedef BaseImageVector::const_iterator const_iterator;

	typedef neuralnetwork::NeuralNetwork NeuralNetwork;

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
	size_t size()  const;
	bool   empty() const;

public:
	void clear();

public:
	void push_back(const Image& image);

public:
	Matrix getDownsampledFeatureMatrix(size_t xTileSize, size_t yTileSize, size_t colors) const;
	Matrix getRandomCropFeatureMatrix(size_t xTileSize, size_t yTileSize, size_t colors, std::default_random_engine& engine, double cropWindowRatio) const;
    Matrix getReference(const util::StringVector& labels) const;

private:
	BaseImageVector _images;

};

}

}


