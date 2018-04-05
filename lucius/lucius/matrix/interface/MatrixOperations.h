
#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;         } }
namespace lucius { namespace matrix { class StaticOperator; } }
namespace lucius { namespace matrix { class Precision;      } }
namespace lucius { namespace matrix { class Dimension;      } }

namespace lucius
{
namespace matrix
{

void apply(Matrix& result, const Matrix& left, const Matrix& right, const StaticOperator& op);
Matrix apply(const Matrix& left, const Matrix& right, const StaticOperator& op);

void apply(Matrix& result, const Matrix& input, const StaticOperator& op);
Matrix apply(const Matrix& input, const StaticOperator& op);

void reduce(Matrix& result, const Matrix& input, const Dimension& d, const StaticOperator& op);
Matrix reduce(const Matrix& input, const Dimension& d, const StaticOperator& op);

void broadcast(Matrix& result, const Matrix& left, const Matrix& right,
    const Dimension& d, const StaticOperator& op);
Matrix broadcast(const Matrix& left, const Matrix& right, const Dimension& d,
    const StaticOperator& op);

void zeros(Matrix& result);
Matrix zeros(const Dimension& size, const Precision& precision);

void ones(Matrix& result);
Matrix ones(const Dimension& size, const Precision& precision);

void range(Matrix& result);
Matrix range(const Dimension& size, const Precision& precision);

void reduceGetPositions(Matrix& result, const Matrix& input, const Dimension& d,
    const StaticOperator& op);
Matrix reduceGetPositions(const Matrix& input, const Dimension& d, const StaticOperator& op);

}
}

