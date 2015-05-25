#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Precision; } }

namespace minerva
{
namespace matrix
{

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

Matrix copy(const Matrix& input, const Precision&);

}
}


