/*	\file   NonDifferentiableLinearSolverFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the NonDifferentiableLinearSolverFactory class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/NonDifferentiableLinearSolverFactory.h>

#include <minerva/optimizer/interface/SimulatedAnnealingSolver.h>

namespace minerva
{

namespace optimizer
{

NonDifferentiableLinearSolver* NonDifferentiableLinearSolverFactory::create
	const std::string& name)
{
	NonDifferentiableLinearSolver* solver = nullptr;
	
	if(name == "SimulatedAnnealingSolver")
	{
		return new SimulatedAnnealingSolver;
	}
	
	return solver;
}

}

}







