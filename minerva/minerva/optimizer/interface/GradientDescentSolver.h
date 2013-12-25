/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Gradient descent Solver class 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropagation.h>

namespace minerva
{
namespace optimizer
{

class GradientDescentSolver : public Solver
{
public:
	GradientDescentSolver(BackPropagation* d) : Solver(d)
	{
	}
	
	void solve();

};

}
}
