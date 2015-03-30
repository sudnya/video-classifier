#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class MatrixVector; } }
namespace minerva { namespace matrix { class Operation;    } }
namespace minerva { namespace matrix { class Precision;    } }

namespace minerva
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


