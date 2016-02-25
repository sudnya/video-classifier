/*  \file   CTCCostFunction.cpp
    \date   Feb 24th, 2016
    \author Sudnya Diamos <mailsudnya@gmail.com>
    \brief  The source file for the CTCCostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/CTCCostFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/debug.h>

#include <string>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;

CTCCostFunction::~CTCCostFunction()
{

}

//void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations, const Matrix& reference);

Matrix CTCCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
    size_t miniBatchSize = output.size()[1];

    Matrix cost({miniBatchSize}, output.precision());
    Matrix fakeGradients();

    matrix::computeCtc(cost, fakeGradients, output, reference);

    return cost;
}

Matrix CTCCostFunction::computeDelta(const Matrix& output, const Matrix& reference) const
{
    size_t miniBatchSize = output.size()[1];

    Matrix cost({miniBatchSize}, output.precision());
    Matrix gradients(output.size(), output.precision());

    matrix::computeCtc(cost, gradients, output, reference);

    return gradients;
}

CostFunction* CTCCostFunction::clone() const
{
    return new CTCCostFunction;
}

std::string CTCCostFunction::typeName() const
{
    return "CTCCostFunction";
}

}

}




