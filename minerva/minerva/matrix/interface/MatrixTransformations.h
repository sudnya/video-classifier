
#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{
namespace matrix
{

Matrix reshape(const Matrix& matrix, const Dimension& );
Matrix flatten(const Matrix& matrix);
 
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end);
Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end, const Dimension& stride);
Matrix resize(const Matrix& input, const Dimension& size);
Matrix reshape(const Matrix& input, const Dimension& size);

Dimension linearStride(const Dimension& );
Dimension zeros(const Dimension& );
Dimension ones(const Dimension& );
Dimension removeDimensions(const Dimension& base, const Dimension& toRemove);
Dimension intersection(const Dimension& base, const Dimension& toRemove);
size_t dotProduct(const Dimension& left, const Dimension& right);
Dimension linearToDimension(size_t linearIndex, const Dimension& size);
Dimension selectNamedDimensions(const Dimension& selectedDimensions, const Dimension& left, const Dimension& right);

void* getAddress(const Dimension& stride, const Dimension& position, void* data, const Precision& precision);
const void* getAddress(const Dimension& stride, const Dimension& position, const void* data, const Precision& precision);

}
}


