#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void fft(Matrix& result, const Matrix& signal, const Dimension& dimensionsToTransform);
Matrix fft(const Matrix& signal, const Dimension& dimensionsToTransform);

}
}


