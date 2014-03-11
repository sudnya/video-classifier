/*	\file   GeneralNondifferentiableSolverFactory.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the GeneralNondifferentiableSolverFactory class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/GeneralNondifferentiableSolverFactory.h>

#include <minerva/optimizer/interface/SimulatedAnnealingSolver.h>

namespace minerva
{

namespace optimizer
{

GeneralNondifferentiableSolver* GeneralNondifferentiableSolverFactory::create(
	const std::string& name)
{
	GeneralNondifferentiableSolver* solver = nullptr;
	
	if(name == "SimulatedAnnealingSolver")
	{
		return new SimulatedAnnealingSolver;
	}
	
	return solver;
}

}

}







