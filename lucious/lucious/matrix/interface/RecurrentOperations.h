#pragma once

// Forward Declarations
namespace lucious { namespace matrix { class Matrix;    } }
namespace lucious { namespace matrix { class Operation; } }

namespace lucious
{
namespace matrix
{

enum RecurrentTimeDirection
{
    RECURRENT_REVERSE_TIME,
    RECURRENT_FORWARD_TIME
};

void forwardRecurrentActivations(Matrix& inputAndOutput, const Matrix& weights, const RecurrentTimeDirection& d, const Operation& activationFunction);
Matrix forwardRecurrentActivations(const Matrix& input, const Matrix& weights, const RecurrentTimeDirection& d, const Operation& activationFunction);

void reverseRecurrentDeltas(Matrix& resultDeltas, const Matrix& weights, const Matrix& activations,
    const RecurrentTimeDirection& d, const Operation& activationDerivativeFunction);
Matrix reverseRecurrentDeltas(const Matrix& deltas, const Matrix& weights, const Matrix& activations,
    const RecurrentTimeDirection& d, const Operation& activationDerivativeFunction);

void reverseRecurrentGradients(Matrix& gradients, const Matrix& inputs, const Matrix& deltas, const RecurrentTimeDirection& d);
Matrix reverseRecurrentGradients(const Matrix& inputs, const Matrix& deltas, const RecurrentTimeDirection& d);


}
}


