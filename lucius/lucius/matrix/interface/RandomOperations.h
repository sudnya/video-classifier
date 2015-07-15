#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Operation; } }
namespace lucius { namespace matrix { class Precision; } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void srand(size_t seed);

void rand(Matrix& result);
void randn(Matrix& result);

Matrix rand(const Dimension&, const Precision& );
Matrix randn(const Dimension&, const Precision& );

}
}

