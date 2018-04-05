
#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class StaticOperator; } }

namespace lucius
{
namespace matrix
{

Matrix applyToAdjacentElements(const Matrix& input, size_t dimensionToApplyTo,
    const StaticOperator& op, double initialValue = 0.0);
void applyToAdjacentElements(Matrix& output, const Matrix& input,
    size_t dimensionToApplyTo, const StaticOperator& op, double initialValue = 0.0);

}
}


