/*    \file   ImageVector.cpp
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Vector class.
*/

// Minerva Includes
#include <minerva/video/interface/ImageVector.h>

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

ImageVector::Matrix ImageVector::getDownsampledFeatureMatrix(size_t xTileSize, size_t yTileSize, size_t colors) const
{
    size_t images = _images.size();

    Matrix matrix({xTileSize, yTileSize, colors, images, 1});

    size_t offset = 0;
    for(auto& image : _images)
    {
        auto sample = image.downsample(xTileSize, yTileSize, colors);

        for(size_t c = 0; c < colors; ++c)
        {
            for(size_t y = 0; y < yTileSize; ++y)
            {
                for(size_t x = 0; x < xTileSize; ++x)
                {
                    matrix(x, y, c, offset, 0) = sample.getComponentAt(x, y, c);
                }
            }
        }

        ++offset;
    }

    util::log("ImageVector") << "Input image:" << matrix.toString();

    return matrix;
}

ImageVector::Matrix ImageVector::getRandomCropFeatureMatrix(size_t xTileSize, size_t yTileSize,
    size_t colors, std::default_random_engine& randomEngine) const
{
    size_t images = _images.size();

    Matrix matrix({xTileSize, yTileSize, colors, images, 1});

    size_t offset = 0;
    for(auto& image : _images)
    {
        size_t sampleXTileSize = std::min(xTileSize, image.x());
        size_t sampleYTileSize = std::min(yTileSize, image.y());
        size_t sampleColors    = std::min(colors,    image.colorComponents());

        //size_t xRemainder = image.x() - sampleXTileSize;
        //size_t yRemainder = image.y() - sampleYTileSize;

        size_t xOffset = 0;//std::uniform_int_distribution<size_t>(0, xRemainder)(randomEngine);
        size_t yOffset = 0;//std::uniform_int_distribution<size_t>(0, yRemainder)(randomEngine);

        auto sample = image.getTile(xOffset, yOffset, sampleXTileSize, sampleYTileSize, sampleColors);

        for(size_t c = 0; c < sampleColors; ++c)
        {
            for(size_t y = 0; y < sampleYTileSize; ++y)
            {
                for(size_t x = 0; x < sampleXTileSize; ++x)
                {
                    matrix(x, y, c, offset, 0) = sample.getComponentAt(x, y, c);
                }
            }
        }

        ++offset;
    }

    if(util::isLogEnabled("ImageVector::Detail"))
    {
        util::log("ImageVector::Detail") << "Input image:" << matrix.toString();
    }

    return matrix;
}

ImageVector::Matrix ImageVector::getReference(const util::StringVector& labels) const
{
    Matrix reference(matrix::Dimension({labels.size(), size(), 1}));

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
                reference(outputNeuron, imageId, 0) = 1.0;
            }
            else
            {
                reference(outputNeuron, imageId, 0) = 0.0;
            }
        }
    }

    util::log("ImageVector") << " Generated matrix: " << reference.toString() << "\n";

    return reference;
}

}

}


