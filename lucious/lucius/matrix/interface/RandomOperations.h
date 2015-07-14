#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace lucious { namespace matrix { class Matrix;    } }
namespace lucious { namespace matrix { class Operation; } }
namespace lucious { namespace matrix { class Precision; } }
namespace lucious { namespace matrix { class Dimension; } }

namespace lucious
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

