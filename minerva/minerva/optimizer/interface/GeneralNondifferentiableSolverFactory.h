/*	\file   GeneralNondifferentiableSolverFactory.h
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the GeneralNondifferentiableSolverFactory class.
*/

#pragma once

// Standard Library Includes
#include <string>

// Forward Declarations
namespace minerva
{
	namespace optimizer { class GeneralNondifferentiableSolver; }
}

namespace minerva
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







