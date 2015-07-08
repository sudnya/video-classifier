/*	\file   GeneralNondifferentiableSolverFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GeneralNondifferentiableSolverFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace lucious
{
	namespace optimizer { class GeneralNondifferentiableSolver; }
}

namespace lucious
{

namespace optimizer
{

/*! \brief A factory for linear optimizers */
class GeneralNondifferentiableSolverFactory
{
public:
	static GeneralNondifferentiableSolver* create(const std::string& name);

};

}

}







