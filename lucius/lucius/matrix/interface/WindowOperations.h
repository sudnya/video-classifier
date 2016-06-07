#pragma once

#include <cstddef>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void hanningWindow(Matrix& result, const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize);
Matrix hanningWindow(const Matrix& signal, const Dimension& dimensionsToTransform, const size_t windowSize);

}
}

