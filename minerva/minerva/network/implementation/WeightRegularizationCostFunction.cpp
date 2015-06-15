/*    \file   WeightRegularizationCostFunction.cpp
    \date   November 19, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the WeightRegularizationCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/WeightRegularizationCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/Knobs.h>

namespace minerva
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

}

}



