#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Dimension; } }

namespace lucius
{
namespace matrix
{

Matrix forwardMaxPooling(const Matrix& input, const Dimension& poolingSize);
void forwardMaxPooling(Matrix& result, const Matrix& input, const Dimension& poolingSize);

Matrix backwardMaxPooling(const Matrix& inputActivations, const Matrix& outputActivations,
    const Matrix& outputDeltas, const Dimension& poolingSize);
void backwardMaxPooling(Matrix& result, const Matrix& inputActivations,
    const Matrix& outputActivations, const Matrix& outputDeltas, const Dimension& poolingSize);

}
}


