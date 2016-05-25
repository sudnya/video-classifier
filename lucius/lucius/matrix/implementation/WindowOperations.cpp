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
void hanningWindow(Matrix& result, const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize)
{
    assert (dimensionsToTransform.size() == 1);
    auto frameSize        = signal.size()[0];
    auto outputFrameSize  = signal.size()[1];
    
    HanningGather hg(signal.size(), signal.stride(), result.size());
    gather(result, signal, hg);

    double pi = 2.0 * std::acos(0.0);

    auto hanningWindow = range({outputFrameSize}, result.precision());
    apply(hanningWindow, hanningWindow, Multiply(2*pi / (outputFrameSize - 1)));
    apply(hanningWindow, hanningWindow, Cos());
    apply(hanningWindow, hanningWindow, Multiply(-1/2.0));
    apply(hanningWindow, hanningWindow, Add(1/2.0));

    broadcast(result, hanningWindow, result, Multiply());
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
