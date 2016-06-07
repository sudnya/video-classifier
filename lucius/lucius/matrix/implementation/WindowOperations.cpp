/*  \file   WindowOperations.cpp
    \date   Monday May 23, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the Window operation (Hanning)
*/

// Lucius Includes
#include <lucius/matrix/interface/WindowOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/Matrix.h>

// Standard Library Includes
#include <cmath>

namespace lucius
{

namespace matrix
{
void hanningWindow(Matrix& result, const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize)
{
    assert(dimensionsToTransform.size() == 1);
    assert(signal.size().size() == 3);

    auto outputFrameSize  = signal.size()[0];
    
    HanningGather hg(signal.size(), signal.stride(), result.size());
    gather(result, signal, hg);

    double pi = 2.0 * std::acos(0.0);

    auto window = range({outputFrameSize, signal.size()[1], signal.size()[2]}, result.precision());
    apply(window, window, Multiply(2*pi / (outputFrameSize - 1)));
    apply(window, window, Cos());
    apply(window, window, Multiply(-1/2.0));
    apply(window, window, Add(1/2.0));

    broadcast(result, window, result, {}, Multiply());
}


Matrix hanningWindow(const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize)
{
    Dimension outputDimensions = signal.size();
    for (auto i : dimensionsToTransform)
    {
        outputDimensions[i] = outputDimensions[i]*windowSize;
    }
    Matrix retVal(outputDimensions, signal.precision());
    hanningWindow(retVal, signal, dimensionsToTransform, windowSize);
    return retVal;
}

}
}
