
#pragma once

// Forward Declarations
namespace lucious { namespace matrix { class Matrix;    } }
namespace lucious { namespace matrix { class Dimension; } }

namespace lucious
{
namespace matrix
{

Dimension forwardConvolutionOutputSize(const Dimension& inputSize, const Dimension& filterSize, const Dimension& filterStride, const Dimension& padding);

void forwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding);
Matrix forwardConvolution(const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding);

void reverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter, const Dimension& stride, const Matrix& deltas, const Dimension& padding);

void reverseConvolutionGradients(Matrix& gradients, const Matrix& inputs, const Matrix& deltas, const Dimension& stride, const Dimension& padding);

void reverseConvolutionGradients(Matrix& gradients, const Matrix& inputs, const Matrix& deltas, const Dimension& stride, const Dimension& padding, double alpha);

}
}


