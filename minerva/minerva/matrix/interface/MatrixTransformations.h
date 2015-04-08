
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
Dimension zeros(const Dimension& );
Dimension removeDimensions(const Dimension& base, const Dimension& toRemove);

void* getAddress(const Dimension& stride, const Dimension& position, void* data, const Precision& precision);
const void* getAddress(const Dimension& stride, const Dimension& position, const void* data, const Precision& precision);

}
}


