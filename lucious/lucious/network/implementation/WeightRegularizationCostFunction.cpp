/*    \file   WeightRegularizationCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the WeightRegularizationCostFunction class.
*/

// Lucious Includes
#include <lucious/network/interface/WeightRegularizationCostFunction.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/Operation.h>

#include <lucious/util/interface/Knobs.h>

namespace lucious
{

namespace network
{

WeightRegularizationCostFunction::~WeightRegularizationCostFunction()
{

}

double WeightRegularizationCostFunction::getCost(const Matrix& weights) const
{
    double lambda = util::KnobDatabase::getKnobValue("NeuralNetwork::Lambda", 0.000);

    return reduce(apply(weights, matrix::Square()), {}, matrix::Add())[0] * (lambda / 2.0);
}

WeightRegularizationCostFunction::Matrix WeightRegularizationCostFunction::getGradient(const Matrix& weights) const
{
    double lambda = util::KnobDatabase::getKnobValue("NeuralNetwork::Lambda", 0.000);

    return apply(weights, matrix::Multiply(lambda));
}

WeightCostFunction* WeightRegularizationCostFunction::clone() const
{
    return new WeightRegularizationCostFunction;
}

std::string WeightRegularizationCostFunction::typeName() const
{
    return "WeightRegularizationCostFunction";
}

}

}



