
#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{
namespace matrix
{

Matrix softmax(const Matrix& input);
Matrix softmaxGradient(const Matrix& output, const Matrix& outputGradient);

Matrix logsoftmax(const Matrix& input);
Matrix logsoftmaxGradient(const Matrix& output, const Matrix& outputGradient);

}
}


