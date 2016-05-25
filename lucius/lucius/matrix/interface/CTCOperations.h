#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{
namespace matrix
{

typedef std::vector<std::vector<size_t>> LabelVector;
typedef std::vector<size_t> IndexVector;

void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations,
    const LabelVector& labels, const IndexVector& timestepsPerSample);
}
}


