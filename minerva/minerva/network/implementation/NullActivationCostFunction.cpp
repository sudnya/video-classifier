/*	\file   NullActivationCostFunction.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NullActivationCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/NullActivationCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace network
{

NullActivationCostFunction::~NullActivationCostFunction()
{

}

float NullActivationCostFunction::getCost(const Matrix& activations) const
{
	return 0.0f;
}

NullActivationCostFunction::Matrix NullActivationCostFunction::getGradient(const Matrix& activations) const
{
	assertM(false, "Not implemented");
}

ActivationCostFunction* NullActivationCostFunction::clone() const
{
	return new NullActivationCostFunction(*this);
}

}

}


