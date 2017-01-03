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
#include <lucius/matrix/interface/RecurrentOperations.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace network
{

typedef matrix::MatrixVector MatrixVector;
typedef matrix::Matrix       Matrix;
typedef std::vector<std::vector<size_t>> LabelVector;
typedef matrix::IndexVector IndexVector;

SumOfSquaresCostFunction::~SumOfSquaresCostFunction()
{

}

void SumOfSquaresCostFunction::computeCostImplementation(Bundle& bundle) const
{
    auto& output    = bundle[   "outputActivations"].get<MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<MatrixVector>().front();

    auto difference = apply(Matrix(output), reference, matrix::Subtract());

    size_t samples = output.size()[output.size().size() - 2];

    auto costs = apply(difference, matrix::SquareAndScale(0.5 / samples));

    if(util::isLogEnabled("SumOfSquaresCostFunction::Detail"))
    {
        util::log("SumOfSquaresCostFunction::Detail") << " costs : "
            << costs.debugString();
        util::log("SumOfSquaresCostFunction::Detail") << " samples : "
            << samples << "\n";
    }

    bundle["costs"] = costs;
}

void SumOfSquaresCostFunction::computeDeltaImplementation(Bundle& bundle) const
{
    auto& output    = bundle[   "outputActivations"].get<MatrixVector>().front();
    auto& reference = bundle["referenceActivations"].get<MatrixVector>().front();

    bundle["outputDeltas"] = MatrixVector({apply(Matrix(output), reference, matrix::Subtract())});
}

std::unique_ptr<CostFunction> SumOfSquaresCostFunction::clone() const
{
    return std::make_unique<SumOfSquaresCostFunction>();
}

std::string SumOfSquaresCostFunction::typeName() const
{
    return "SumOfSquaresCostFunction";
}

}

}

