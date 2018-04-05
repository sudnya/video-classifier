#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;          } }
namespace lucius { namespace matrix { class Precision;       } }
namespace lucius { namespace matrix { class Dimension;       } }
namespace lucius { namespace matrix { class StaticOperator;  } }

namespace lucius
{
namespace matrix
{

void gather(Matrix& result, const Matrix& input, const StaticOperator& op);
Matrix gather(const Matrix& input, const StaticOperator& op);

void permuteDimensions(Matrix& result, const Matrix& input, const Dimension& newOrder);
Matrix permuteDimensions(const Matrix& input, const Dimension& newOrder);

void indirectGather(Matrix& result, const Matrix& input, const Matrix& indices,
    const StaticOperator& mapper);

}
}



