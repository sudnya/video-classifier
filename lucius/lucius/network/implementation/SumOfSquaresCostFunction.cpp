/*  \file   SumOfSquaresCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SumOfSquaresCostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/SumOfSquaresCostFunction.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>

namespace lucius
{

namespace network
{

typedef matrix::MatrixVector MatrixVector;
typedef matrix::Matrix       Matrix;

SumOfSquaresCostFunction::~SumOfSquaresCostFunction()
{

}

void SumOfSquaresCostFunction::computeCost(Bundle& bundle) const
{
    auto& output    = bundle[   "outputActivations"].get<MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<MatrixVector>().front();

    auto difference = apply(Matrix(output), reference, matrix::Subtract());

    size_t samples = output.size()[output.size().size() - 2];

    bundle["costs"] = apply(difference, matrix::SquareAndScale(0.5 / samples));
}

void SumOfSquaresCostFunction::computeDelta(Bundle& bundle) const
{
    auto& output    = bundle[   "outputActivations"].get<MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<MatrixVector>().front();

    bundle["outputDeltas"] = MatrixVector({apply(Matrix(output), reference, matrix::Subtract())});
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

