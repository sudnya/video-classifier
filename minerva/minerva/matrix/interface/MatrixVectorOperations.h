#pragma once

// Forward Declarations
namespace lucious { namespace matrix { class MatrixVector; } }
namespace lucious { namespace matrix { class Operation;    } }
namespace lucious { namespace matrix { class Precision;    } }
namespace lucious { namespace matrix { class Dimension;    } }

namespace lucious
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

void zeros(MatrixVector& result);

}
}


