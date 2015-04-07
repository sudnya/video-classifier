
#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{
namespace matrix
{

Matrix reshape(const Matrix& matrix, const Dimension& );
Matrix flatten(const Matrix& matrix);

Dimension linearStride(const Dimension& );
Dimension removeDimensions(const Dimension& base, const Dimension& toRemove);

}
}


