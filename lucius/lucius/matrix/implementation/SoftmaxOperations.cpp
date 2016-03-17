
// Lucius Includes
#include <lucius/matrix/interface/SoftmaxOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Matrix.h>

namespace lucius
{
namespace matrix
{

Matrix softmax(const Matrix& output)
{
    auto normalizedOutput = broadcast(output,
        reduce(output, {0}, matrix::Maximum()), {0}, matrix::Subtract());

    auto expOutput = apply(normalizedOutput, matrix::Exp());

    auto sums = reduce(expOutput, {0}, matrix::Add());

    return broadcast(expOutput, sums, {0}, matrix::Divide());
}

}
}


