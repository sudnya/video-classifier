/*	\file   GeneralDifferentiableSolverFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the GeneralDifferentiableSolverFactory class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolver.h>
#include <minerva/optimizer/interface/GPULBFGSSolver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>

#include <minerva/util/interface/Knobs.h>

namespace minerva
{

namespace optimizer
{

GeneralDifferentiableSolver* GeneralDifferentiableSolverFactory::create(const std::string& name)
{
	GeneralDifferentiableSolver* solver = nullptr;
	
	if("LimitedMemoryBroydenFletcherGoldfarbShannoSolver" == name ||
		"LBFGSSolver" == name)
	{
		if(GPULBFGSSolver::isSupported())
		{
			solver = new GPULBFGSSolver;
		}
		else if(LBFGSSolver::isSupported())
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

static std::string getSolverName()
{
	return util::KnobDatabase::getKnobValue("GeneralDifferentiableSolver::Type",
		"LBFGSSolver");
}

GeneralDifferentiableSolver* GeneralDifferentiableSolverFactory::create()
{
	auto solverName = getSolverName();
	
	return create(solverName);
}

double GeneralDifferentiableSolverFactory::getMemoryOverheadForSolver(const std::string& name)
{
	if("LimitedMemoryBroydenFletcherGoldfarbShannoSolver" == name ||
		"LBFGSSolver" == name)
	{
		if(LBFGSSolver::isSupported())
		{
			return LBFGSSolver::getMemoryOverhead();
		}
	}
	else if("GradientDescentSolver" == name || "GDSolver" == name)
	{
		return GDSolver::getMemoryOverhead();
	}
	
	return 2.0;
}

double GeneralDifferentiableSolverFactory::getMemoryOverheadForSolver()
{
	auto solverName = getSolverName();
	 
	return getMemoryOverheadForSolver(solverName);
}

GeneralDifferentiableSolverFactory::StringVector GeneralDifferentiableSolverFactory::enumerate()
{
	return 
	{
		"LimitedMemoryBroydenFletcherGoldfarbShannoSolver",
		"GradientDescentSolver"
	};
}

}

}

