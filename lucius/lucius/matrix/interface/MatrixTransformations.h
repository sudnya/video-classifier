
#pragma once

// Lucius Includes
#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{
namespace matrix
{

Matrix reshape(const Matrix& matrix, const Dimension& );
Matrix flatten(const Matrix& matrix);

Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end);
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(const Matrix& input, const Dimension& size);
Matrix reshape(const Matrix& input, const Dimension& size);

}
}


