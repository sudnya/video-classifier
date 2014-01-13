/*	\file   LinearSolverFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LinearSolverFactory class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/LinearSolverFactory.h>
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>

#include <minerva/util/interface/Knobs.h>

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

static std::string getSolverName()
{
	return util::KnobDatabase::getKnobValue("LinearSolver::Type",
		//"LBFGSSolver");
		"GradientDescentSolver");	
}

LinearSolver* LinearSolverFactory::create()
{
	auto solverName = getSolverName();
	
	return create(solverName);
}

double LinearSolverFactory::getMemoryOverheadForSolver(const std::string& name)
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

double LinearSolverFactory::getMemoryOverheadForSolver()
{
	auto solverName = getSolverName();
	 
	return getMemoryOverheadForSolver(solverName);
}

}

}

