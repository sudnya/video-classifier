/*  \file   InputCameraDataProducer.cpp
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputCameraDataProducer class.
*/

// Lucius Includes
#include <lucius/input/interface/InputCameraDataProducer.h>

#include <lucius/video/interface/Camera.h>
#include <lucius/video/interface/Image.h>
#include <lucius/video/interface/ImageVector.h>

#include <lucius/matrix/interface/Matrix.h>

// Standard Library Includes
#include <limits>

namespace lucius
{

namespace input
{

typedef video::ImageVector ImageVector;

InputCameraDataProducer::InputCameraDataProducer()
: _camera(new Camera)
{

}

InputCameraDataProducer::~InputCameraDataProducer()
{

}

void InputCameraDataProducer::initialize()
{
    // intentionally blank
}

InputCameraDataProducer::InputAndReferencePair InputCameraDataProducer::pop()
{
    auto imageDimension = getInputSize();

    auto image = _camera->popFrame();

    image.displayOnScreen();

    ImageVector batch;

    batch.push_back(image);

    auto input = batch.getDownsampledFeatureMatrix(imageDimension[0], imageDimension[1],
        imageDimension[2]);

    standardize(input);

    return InputAndReferencePair(std::move(input), Matrix());
}

bool InputCameraDataProducer::empty() const
{
    return false;
}

void InputCameraDataProducer::reset()
{
    // intentionally blank
}

size_t InputCameraDataProducer::getUniqueSampleCount() const
{
    return std::numeric_limits<size_t>::max();
}

}

}



