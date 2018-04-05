#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class MatrixVector;      } }
namespace lucius { namespace matrix { class StaticOperator;    } }
namespace lucius { namespace matrix { class Precision;         } }
namespace lucius { namespace matrix { class Dimension;         } }

namespace lucius
{
namespace matrix
{

void apply(MatrixVector& result, const MatrixVector& left, const MatrixVector& right,
    const StaticOperator& op);
MatrixVector apply(const MatrixVector& left, const MatrixVector& right, const StaticOperator& op);

void apply(MatrixVector& result, const MatrixVector& input, const StaticOperator& op);
MatrixVector apply(const MatrixVector& input, const StaticOperator& op);

void reduce(MatrixVector& result, const MatrixVector& input, const Dimension& d,
    const StaticOperator& op);
MatrixVector reduce(const MatrixVector& input, const Dimension& d, const StaticOperator& op);

double dotProduct(const MatrixVector& left, const MatrixVector& right);

void copy(MatrixVector& result, const MatrixVector& input);
MatrixVector copy(const MatrixVector& input);

void zeros(MatrixVector& result);

}
}


