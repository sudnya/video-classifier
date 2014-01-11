/*	\file   LinearSolverFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LinearSolverFactory class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/LinearSolverFactory.h>
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>

namespace minerva
{

namespace optimizer
{

LinearSolver* LinearSolverFactory::create(const std::string& name)
{
	LinearSolver* solver = nullptr;
	
	if("LimitedMemoryBroydenFletcherGoldfarbShannoSolver" == name ||
		"LBFGSSolver" == name)
	{
		if(LBFGSSolver::isSupported())
		{
			solver = new LBFGSSolver;
		}
	}
	else if("GradientDescentSolver" == name || "GDSolver" == name)
	{
		solver = new GDSolver;
	}
	
	return solver;
}

}

}

