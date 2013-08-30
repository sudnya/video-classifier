/*	\file   ImageVector.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Vector class.
*/

// Minerva Includes
#include <minerva/video/interface/ImageVector.h>

#include <minerva/neuralnetwork/interface/NeuralNetwork.h>

#include <minerva/util/interface/debug.h>

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

void ImageVector::push_back(const Image& image)
{
	_images.push_back(image);
}

ImageVector::Matrix ImageVector::convertToMatrix(size_t sampleCount) const
{
	size_t rows    = _images.size();
	size_t columns = sampleCount;
	
	Matrix matrix(rows, columns);
	
    size_t row = 0;
	for(auto& image : _images)
	{
		matrix.setRowData(row++, image.getSampledData(columns));
	}
	
	return matrix;
}

ImageVector::Matrix ImageVector::getReference(
	const NeuralNetwork& neuralNetwork) const
{
	Matrix matrix(neuralNetwork.getOutputCount(), size());
	
	for(unsigned int imageId = 0; imageId != size(); ++imageId)
	{
		for(unsigned int outputNeuron = 0;
			outputNeuron != neuralNetwork.getOutputCount(); ++outputNeuron)
		{
			if((*this)[imageId].label() ==
				neuralNetwork.getLabelForOutputNeuron(outputNeuron))
			{
				matrix(outputNeuron, imageId) = 1.0f;
			}
			else
			{
				matrix(outputNeuron, imageId) = 0.0f;
			}
		}
	}
	
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


