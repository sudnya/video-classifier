/*	\file   GeneralDifferentiableSolverFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GeneralDifferentiableSolverFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace optimizer { class GeneralDifferentiableSolver; } }

namespace minerva
{

namespace optimizer
{

/*! \brief A factory for linear optimizers */
class GeneralDifferentiableSolverFactory
{
public:
	static GeneralDifferentiableSolver* create(const std::string& solverName);

public:
	static GeneralDifferentiableSolver* create();

public:
	static double getMemoryOverheadForSolver(const std::string& solverName);
	static double getMemoryOverheadForSolver();

};

}

}







