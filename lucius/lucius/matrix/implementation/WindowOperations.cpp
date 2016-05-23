/*  \file   WindowOperations.cpp
    \date   Monday May 23, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the Window operation (Hanning)
*/

// Lucius Includes
#include <lucius/matrix/interface/WindowOperations.h>

// Standard Library Includes

// System-Specific Includes

namespace lucius
{

namespace matrix
{

void hanningWindow(Matrix& result, const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize)
{
    
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
