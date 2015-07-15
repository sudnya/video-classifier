#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{
namespace matrix
{

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

Matrix copy(const Matrix& input, const Precision&);

}
}


