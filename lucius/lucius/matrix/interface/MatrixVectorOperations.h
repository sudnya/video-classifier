#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class MatrixVector; } }
namespace lucius { namespace matrix { class Operation;    } }
namespace lucius { namespace matrix { class Precision;    } }
namespace lucius { namespace matrix { class Dimension;    } }

namespace lucius
{
namespace matrix
{

void apply(MatrixVector& result, const MatrixVector& left, const MatrixVector& right, const Operation& op);
MatrixVector apply(const MatrixVector& left, const MatrixVector& right, const Operation& op);

void apply(MatrixVector& result, const MatrixVector& input, const Operation& op);
MatrixVector apply(const MatrixVector& input, const Operation& op);

void reduce(MatrixVector& result, const MatrixVector& input, const Dimension& d, const Operation& op);
MatrixVector reduce(const MatrixVector& input, const Dimension& d, const Operation& op);

double dotProduct(const MatrixVector& left, const MatrixVector& right);

void copy(MatrixVector& result, const MatrixVector& input);
MatrixVector copy(const MatrixVector& input);

void zeros(MatrixVector& result);

}
}


