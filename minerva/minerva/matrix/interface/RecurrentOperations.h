#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Operation; } }

namespace minerva
{
namespace matrix
{

enum RecurrentDirection
{
    RECURRENT_REVERSE,
    RECURRENT_FORWARD
};

void forwardRecurrent(Matrix& result, const Matrix& input, const Matrix& weights, const RecurrentDirection& d, const Operation& activationFunction);
Matrix forwardRecurrent(const Matrix& input, const Matrix& weights, const RecurrentDirection& d, const Operation& activationFunction);

void reverseRecurrentDeltas(Matrix& resultDeltas, const Matrix& weights, const Matrix& deltas);
Matrix reverseRecurrentDeltas(const Matrix& weights, const Matrix& deltas);

void reverseRecurrentGradients(Matrix& gradients, const Matrix& weights, const Matrix& inputs, const Matrix& deltas);
Matrix reverseRecurrentGradients(const Matrix& weights, const Matrix& inputs, const Matrix& deltas);


}
}


