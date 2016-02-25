#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }

namespace lucius
{
namespace matrix
{
void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations, const Matrix& reference);
}
}


