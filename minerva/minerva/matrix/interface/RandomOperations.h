#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Operation; } }
namespace minerva { namespace matrix { class Precision; } }

namespace minerva
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

