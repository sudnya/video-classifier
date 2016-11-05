
#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Operation; } }

namespace lucius
{
namespace matrix
{

Matrix reduceByKey(const Matrix& keys, const Matrix& values, const Operation& op);
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values, const Operation& op);

}
}

