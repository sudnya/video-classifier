/*  \file   InputOutputLearnerEngine.cpp
    \date   January 16, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the InputOutputLearnerEngine class.
*/

// Lucius Includes
#include <lucius/engine/interface/InputOutputLearnerEngine.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>

#include <lucius/results/interface/ResultVector.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace engine
{

typedef matrix::Dimension Dimension;
typedef matrix::Matrix Matrix;
typedef results::ResultVector ResultVector;

InputOutputLearnerEngine::InputOutputLearnerEngine()
{

}

InputOutputLearnerEngine::~InputOutputLearnerEngine()
{

}

Dimension InputOutputLearnerEngine::getInputDimensions() const
{
    Dimension inputDimension;

    assertM(false, "Not implemented.");

    return inputDimension;
}

Matrix InputOutputLearnerEngine::getInputSlice(Dimension begin, Dimension end) const
{
    Matrix slice;

    assertM(false, "Not implemented.");

    return slice;
}

Dimension InputOutputLearnerEngine::getOutputDimensions() const
{
    Dimension inputDimension;

    assertM(false, "Not implemented.");

    return inputDimension;
}

Matrix InputOutputLearnerEngine::getOutputSlice(Dimension begin, Dimension end) const
{
    Matrix slice;

    assertM(false, "Not implemented.");

    return slice;
}

Dimension InputOutputLearnerEngine::getDimensions() const
{
    Dimension inputDimension;

    assertM(false, "Not implemented.");

    return inputDimension;
}

Matrix InputOutputLearnerEngine::getSlice(Dimension begin, Dimension end) const
{
    Matrix slice;

    assertM(false, "Not implemented.");

    return slice;
}

ResultVector InputOutputLearnerEngine::runOnBatch(Bundle& bundle)
{
    ResultVector result;

    assertM(false, "Not implemented.");

    return result;
}

bool InputOutputLearnerEngine::requiresLabeledData() const
{
    return true;
}

}

}






