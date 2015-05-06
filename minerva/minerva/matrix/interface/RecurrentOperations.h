#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix;    } }
namespace minerva { namespace matrix { class Operation; } }

namespace minerva
{
namespace matrix
{

enum RecurrentTimeDirection
{
    RECURRENT_REVERSE_TIME,
    RECURRENT_FORWARD_TIME
};

void forwardRecurrent(Matrix& result, const Matrix& input, const Matrix& weights, const RecurrentTimeDirection& d, const Operation& activationFunction);
Matrix forwardRecurrent(const Matrix& input, const Matrix& weights, const RecurrentTimeDirection& d, const Operation& activationFunction);

void reverseRecurrentDeltas(Matrix& resultDeltas, const Matrix& weights, const Matrix& deltas, const RecurrentTimeDirection& d);
Matrix reverseRecurrentDeltas(const Matrix& weights, const Matrix& deltas, const RecurrentTimeDirection& d);

void reverseRecurrentGradients(Matrix& gradients, const Matrix& weights, const Matrix& inputs, const Matrix& deltas, const RecurrentTimeDirection& d);
Matrix reverseRecurrentGradients(const Matrix& weights, const Matrix& inputs, const Matrix& deltas, const RecurrentTimeDirection& d);


}
}


