/*  \file   SumOfSquaresCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SumOfSquaresCostFunction class.
*/

// Lucious Includes
#include <lucious/network/interface/SumOfSquaresCostFunction.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/Operation.h>

namespace lucious
{

namespace network
{

SumOfSquaresCostFunction::~SumOfSquaresCostFunction()
{

}

matrix::Matrix SumOfSquaresCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
    auto difference = apply(output, reference, matrix::Subtract());

    size_t samples = output.size()[output.size().size() - 2];

    return apply(difference, matrix::SquareAndScale(0.5/samples));
}

matrix::Matrix SumOfSquaresCostFunction::computeDelta(const Matrix& output, const Matrix& reference) const
{
    return apply(output, reference, matrix::Subtract());
}

CostFunction* SumOfSquaresCostFunction::clone() const
{
    return new SumOfSquaresCostFunction;
}

std::string SumOfSquaresCostFunction::typeName() const
{
    return "SumOfSquaresCostFunction";
}

}

}

