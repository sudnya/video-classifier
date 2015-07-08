/*	\file   GeneralDifferentiableSolverFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GeneralDifferentiableSolverFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace lucious { namespace optimizer { class GeneralDifferentiableSolver; } }

namespace lucious
{

namespace optimizer
{

/*! \brief A factory for differentiable optimizers */
class GeneralDifferentiableSolverFactory
{
public:
	typedef std::vector<std::string> StringVector;

public:
	static GeneralDifferentiableSolver* create(const std::string& solverName);

public:
	static GeneralDifferentiableSolver* create();

public:
	static double getMemoryOverheadForSolver(const std::string& solverName);
	static double getMemoryOverheadForSolver();

public:
	static StringVector enumerate();

};

}

}







