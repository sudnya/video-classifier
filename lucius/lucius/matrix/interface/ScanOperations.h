#pragma once

// Standard Library Includes
#include <cstddef>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }
namespace lucius { namespace matrix { class Operation; } }

namespace lucius
{
namespace matrix
{

void inclusiveScan(Matrix& output, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue);
Matrix inclusiveScan(const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue);

void exclusiveScan(Matrix& output, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue);
Matrix exclusiveScan(const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue);


}
}


