
// Lucius Includes
#include <lucius/matrix/interface/SoftmaxOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{
namespace matrix
{

Matrix softmax(const Matrix& input)
{
    auto normalizedInput = broadcast(input,
        reduce(input, {0}, matrix::Maximum()), {0}, matrix::Subtract());

    auto expOutput = apply(normalizedInput, matrix::Exp());

    auto sums = reduce(expOutput, {0}, matrix::Add());

    return broadcast(expOutput, sums, {0}, matrix::Divide());
}

Matrix softmaxGradient(const Matrix& output, const Matrix& outputGradient)
{
    auto sum = reduce(apply(matrix::Matrix(output), outputGradient, Multiply()),
        {0}, Add());

    return apply(broadcast(outputGradient, sum, {0}, Subtract()), output, Multiply());
}

Matrix logsoftmax(const Matrix& input)
{
    auto normalizedOutput = broadcast(input,
        reduce(input, {0}, matrix::Maximum()), {0}, Subtract());

    auto expOutput = apply(normalizedOutput, Exp());

    auto sums = reduce(expOutput, {0}, Add());

    return broadcast(normalizedOutput, apply(sums, Log()), {0}, Subtract());
}

Matrix logsoftmaxGradient(const Matrix& input, const Matrix& outputGradient)
{
    auto sum = reduce(outputGradient, {0}, matrix::Add());

    return apply(outputGradient,
        broadcast(apply(input, Exp()), sum, {0}, Multiply()),
        Subtract());
}

}
}


