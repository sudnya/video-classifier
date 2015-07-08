/*! \file   Solver.h
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Saturday March 8, 2014
	\brief  The header file for the Solver class.
*/

#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace lucious { namespace matrix    { class Matrix;     } }
namespace lucious { namespace optimizer { class Constraint; } }

namespace lucious
{

namespace optimizer
{

/*! \brief A general interface for an optimization problem solver */
class Solver
{
public:
	Solver();
	virtual ~Solver();

public:
	virtual void addConstraint(const Constraint& constraint);

public:
	Solver(const Solver&) = delete;
	Solver& operator=(const Solver&) = delete;

protected:
	typedef std::vector<Constraint*> ConstraintVector;

protected:
	ConstraintVector _constraints;
	
};

}

}

