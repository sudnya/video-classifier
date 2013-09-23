/*	\file   NonDifferentiableLinearSolverFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the NonDifferentiableLinearSolverFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva
{
	namespace optimizer { class NonDifferentiableLinearSolver; }
}

namespace minerva
{

namespace optimizer
{

/*! \brief A factory for linear optimizers */
class NonDifferentiableLinearSolverFactory
{
public:
	static NonDifferentiableLinearSolver* create(const std::string& name);

};

}

}







