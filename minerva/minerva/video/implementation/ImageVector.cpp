/*	\file   ImageVector.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Vector class.
*/

// Minerva Includes
#include <minerva/video/interface/ImageVector.h>

#include <minerva/network/interface/NeuralNetwork.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/math.h>

namespace minerva
{

namespace video
{

ImageVector::ImageVector()
{

}

ImageVector::~ImageVector()
{

}

ImageVector::iterator ImageVector::begin()
{
	return _images.begin();
}

ImageVector::const_iterator ImageVector::begin() const
{
	return _images.begin();
}


ImageVector::iterator ImageVector::end()
{
	return _images.end();
}

ImageVector::const_iterator ImageVector::end() const
{
	return _images.end();
}

Image& ImageVector::operator[](size_t index)
{
	return _images[index];
}

const Image& ImageVector::operator[](size_t index) const
{
	return _images[index];
}

Image& ImageVector::back()
{
	return _images.back();
}

const Image& ImageVector::back() const
{
	return _images.back();
}

size_t ImageVector::size() const
{
	return _images.size();
}

bool ImageVector::empty() const
{
	return _images.empty();
}

void ImageVector::push_back(const Image& image)
{
	_images.push_back(image);
}

void ImageVector::clear()
{
	_images.clear();
}

ImageVector::Matrix ImageVector::convertToStandardizedMatrix(size_t sampleCount,
	size_t xTileSize, size_t yTileSize, size_t colors) const
{
	size_t rows    = _images.size();
	size_t columns = sampleCount;
	
	Matrix matrix(rows, columns);
	
	Matrix::FloatVector data(rows * columns);
	
    size_t offset = 0;
	for(auto& image : _images)
	{
		assert(colors == image.colorComponents());
		
		auto samples = image.getSampledData(columns, xTileSize, yTileSize);
		
		std::copy(samples.begin(), samples.end(), data.begin() + offset);
		
		offset += columns;
	}
	
	matrix.data() = data;
	
	// remove mean
	matrix = matrix.add(-matrix.reduceSum()/matrix.size());

	// truncate to 3 standard deviations
	float standardDeviation = 3.0f * std::sqrt(matrix.elementMultiply(matrix).reduceSum() / matrix.size());

	matrix.maxSelf(-standardDeviation);
	matrix.minSelf( standardDeviation);

	// scale from [-1,1]
	matrix = matrix.multiply(1.0f/(standardDeviation));

	// rescale from [-1,1] to [0.1, 0.9]
	//matrix = matrix.add(1.0f).multiply(0.4f).add(0.1f);
	
	//std::cout << "Matrix " << matrix.debugString();
	
	return matrix;
}

ImageVector::Matrix ImageVector::convertToStandardizedMatrix(size_t sampleCount, size_t tileSize,
	size_t colors) const
{
	size_t x = 0;
	size_t y = 0;
	
	util::getNearestToSquareFactors(x, y, tileSize / colors);

	return convertToStandardizedMatrix(sampleCount, x, y, colors);
}

ImageVector::Matrix ImageVector::getReference(const util::StringVector& labels) const
{
	Matrix matrix(size(), labels.size());
	
	util::log("ImageVector") << "Generating reference image:\n";
	
	for(unsigned int imageId = 0; imageId != size(); ++imageId)
	{
		util::log("ImageVector") << " For image" << imageId << " with label '"
			<< (*this)[imageId].label() << "'\n";
		
		for(unsigned int outputNeuron = 0;
			outputNeuron != labels.size(); ++outputNeuron)
		{
			util::log("ImageVector") << "  For output neuron" << outputNeuron
				<< " with label '"
				<< labels[outputNeuron] << "'\n";
		
			if((*this)[imageId].label() == labels[outputNeuron])
			{
				matrix(imageId, outputNeuron) = 0.9f;
			}
			else
			{
				matrix(imageId, outputNeuron) = 0.1f;
			}
		}
	}
	
	util::log("ImageVector") << " Generated matrix: " << matrix.toString() << "\n";
	
	return matrix;
}

size_t ImageVector::_getSampledImageSize() const
{
	size_t maxSize = 0;
	
	for(auto& image : _images)
	{
		maxSize = std::max(image.totalSize(), maxSize);
	}
	
	return maxSize;
}

}

}


