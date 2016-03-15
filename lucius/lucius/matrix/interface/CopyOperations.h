#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Precision; } }
namespace lucius { namespace matrix { class Operation; } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

void copy(Matrix& result, const Matrix& input);
Matrix copy(const Matrix& input);

Matrix copy(const Matrix& input, const Precision&);

void gather(Matrix& result, const Matrix& input, const Operation& op);
Matrix gather(const Matrix& input, const Operation& op);

void permuteDimensions(Matrix& result, const Matrix& input, const Dimension& newOrder);
Matrix permuteDimensions(const Matrix& input, const Dimension& newOrder);

}
}


