/*  \file   WindowOperations.cpp
    \date   Monday May 23, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the Window operation (Hanning)
*/

// Lucius Includes
#include <lucius/matrix/interface/WindowOperations.h>

// Standard Library Includes


namespace lucius
{

namespace matrix
{

Dimension convertOutputToInputCoordinate(const Dimension& output, const Dimension& input, const Dimension outputCoordinate)
{
    Dimension retVal = input.size();
    auto frameSize   = input.size[0];
    auto windowSize  = output.size[0]/frameSize;
    
    retVal[0] = outputCoordinate[0]%frameSize;
    retVal[2] = outputCoordinate[2] + outputCoordinate[0]/frameSize;
    
    return retVal;
}

void hanningWindow(Matrix& result, const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize)
{
    assert (dimensionsToTransform.size() == 1);
    auto frameSize        = signal.size()[0];
    auto allOutputCoordinates = getAllOutputCoordinates(result);

    for(auto& outputCoordinate : allOutputCoordinates)
    {
        auto inputCoordinate = convertOutputToInputCoordinate(result.size(), signal.size(), outputCoordinate);

        result[outputCoordinate] = input[inputCoordinate];
    }

}


Matrix hanningWindow(const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize)
{
    Dimension outputDimension = signal.size();
    for (auto i : dimensionsToTransform)
    {
        outputDimensions[i] = outputDimensions[i]*windowSize;
    }
    Matrix retVal(outputDimension, signal.precision());
    hanningWindow(retVal, signal, dimensionsToTransform, windowSize);
    return retVal;
}

}
}
