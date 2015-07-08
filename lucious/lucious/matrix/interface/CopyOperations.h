#pragma once

// Forward Declarations
namespace lucious { namespace matrix { class Matrix;    } }
namespace lucious { namespace matrix { class Precision; } }

namespace lucious
{
namespace matrix
{

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

Matrix copy(const Matrix& input, const Precision&);

}
}


