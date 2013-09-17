/*	\file   LinearSolverFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LinearSolverFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva { namespace optimizer { class LinearSolver; } }

namespace minerva
{

namespace optimizer
{

/*! \brief A factory for linear optimizers */
class LinearSolverFactory
{
public:
	static LinearSolver* create(const std::string& classifierName);

};

}

}







