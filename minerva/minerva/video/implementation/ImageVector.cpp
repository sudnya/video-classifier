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

bool ImageVector::empty() const
{
	return _images.empty();
}

void ImageVector::push_back(const Image& image)
{
	_images.push_back(image);
}

ImageVector::Matrix ImageVector::convertToStandardizedMatrix(size_t sampleCount,
	size_t xTileSize, size_t yTileSize) const
{
	size_t rows    = _images.size();
	size_t columns = sampleCount;
	
	Matrix matrix(rows, columns);
	
	Matrix::FloatVector data(rows * columns);
	
    size_t offset = 0;
	for(auto& image : _images)
	{
		auto samples = image.getSampledData(columns, xTileSize, yTileSize);
		
		std::copy(samples.begin(), samples.end(), data.begin() + offset);
		
		offset += columns;
	}
	
	matrix.data() = data;
	
	return matrix;
}

ImageVector::Matrix ImageVector::getReference(
	const NeuralNetwork& neuralNetwork) const
{
	Matrix matrix(size(), neuralNetwork.getOutputCount());
	
	util::log("ImageVector") << "Generating reference image:\n";
	
	for(unsigned int imageId = 0; imageId != size(); ++imageId)
	{
//		util::log("ImageVector") << " For image" << imageId << " with label '"
//			<< (*this)[imageId].label() << "'\n";
		
		for(unsigned int outputNeuron = 0;
			outputNeuron != neuralNetwork.getOutputCount(); ++outputNeuron)
		{
//			util::log("ImageVector") << "  For output neuron" << outputNeuron
//				<< " with label '"
//				<< neuralNetwork.getLabelForOutputNeuron(outputNeuron) << "'\n";
		
			if((*this)[imageId].label() ==
				neuralNetwork.getLabelForOutputNeuron(outputNeuron))
			{
				matrix(imageId, outputNeuron) = 1.0f;
			}
			else
			{
				matrix(imageId, outputNeuron) = 0.0f;
			}
		}
	}
	
	util::log("ImageVector") << " Generated matrix: " << matrix.toString(10, 20)
		<< "\n";
	
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


