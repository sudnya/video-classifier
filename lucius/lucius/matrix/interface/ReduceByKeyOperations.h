
#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;         } }
namespace lucius { namespace matrix { class Dimension;      } }
namespace lucius { namespace matrix { class StaticOperator; } }

namespace lucius
{
namespace matrix
{

Matrix reduceByKey(const Matrix& keys, const Matrix& values, const Dimension& dimensionsToReduce,
    const StaticOperator& op);
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values,
    const Dimension& dimensionsToReduce, const StaticOperator& op);

void unique(Matrix& output, const Matrix& input, const Dimension& dimensionsToReduce,
    double defaultValue);
Matrix unique(const Matrix& input, const Dimension& dimensionsToReduce, double defaultValue);

}
}

