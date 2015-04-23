
#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Dimension; } }

namespace minerva
{
namespace matrix
{

void forwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter, const Dimension& stride);
Matrix forwardConvolution(const Matrix& input, const Matrix& filter, const Dimension& stride);

void reverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter, const Matrix& deltas);
Matrix reverseConvolutionDeltas(const Matrix& filter, const Matrix& deltas);

void reverseConvolutionGradients(Matrix& gradients, const Matrix& filter, const Matrix& inputs, const Matrix& deltas);
Matrix reverseConvolutionGradients(const Matrix& filter, const Matrix& inputs, const Matrix& deltas);

}
}


