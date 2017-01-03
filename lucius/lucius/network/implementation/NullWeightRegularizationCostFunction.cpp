/*  \file   NullWeightRegularizationCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the NullWeightRegularizationCostFunction class.
*/

// Lucius Includes
#include <lucius/network/interface/NullWeightRegularizationCostFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>

#include <lucius/util/interface/memory.h>

namespace lucius
{

namespace network
{

NullWeightRegularizationCostFunction::~NullWeightRegularizationCostFunction()
{

}

double NullWeightRegularizationCostFunction::getCost(const Matrix& weights) const
{
    return 0.0;
}

NullWeightRegularizationCostFunction::Matrix NullWeightRegularizationCostFunction::getGradient(
    const Matrix& weights) const
{
    return zeros(weights.size(), weights.precision());
}

std::unique_ptr<WeightCostFunction> NullWeightRegularizationCostFunction::clone() const
{
    return std::make_unique<NullWeightRegularizationCostFunction>();
}

std::string NullWeightRegularizationCostFunction::typeName() const
{
    return "NullWeightRegularizationCostFunction";
}

}

}


