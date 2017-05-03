/*  \file   GradientOperations.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the lazy gradient operation interface functions.
*/

// Lucius Includes
#include <lucius/lazy-ir/interface/GradientOperations.h>

namespace lucius
{

namespace lazy
{

VariableAndGradient::VariableAndGradient(const LazyValue& variable, const LazyValue& gradient)
: _variable(variable), _gradient(gradient)
{

}

LazyValue VariableAndGradient::getVariable()
{
    return _variable;
}

LazyValue VariableAndGradient::getGradient()
{
    return _gradient;
}

VariableAndGradientVector getVariablesAndGradientsForCost(const LazyValue& cost)
{
    // Get all variables
    auto variables = builder.getAllVariables();

    // Add gradient variables for each of these
    VariableAndGradientVector variablesAndGradients;

    for(auto& variable : variables)
    {
        auto gradient = builder.addGradientForVariable(variable);

        variablesAndGradients.push_back(VariableAndGradient(variable, gradient));
    }

    return variablesAndGradients;
}

}

}






