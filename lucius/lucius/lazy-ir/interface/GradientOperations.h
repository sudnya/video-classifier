/*  \file   GradientOperations.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the lazy gradient operation interface functions.
*/

#pragma once

// Forward Declarations
class lucius { class lazy { class LazyValue; } }

namespace lucius
{

namespace lazy
{

class VariableAndGradient
{
public:
    VariableAndGradient(const LazyValue& variable, const LazyValue& gradient);

public:
    LazyValue getVariable();
    LazyValue getGradient();

private:
    LazyValue _variable;
    LazyValue _gradient;
};

typedef std::vector<VariableAndGradient> VariableAndGradientVector;

VariableAndGradientVector getVariablesAndGradientsForCost(const LazyValue& cost);

}

}





