/*! \file   Solver.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Saturday March 8, 2014
    \brief  The source file for the Solver class.
*/

// Lucious Includes
#include <lucious/optimizer/interface/Solver.h>
#include <lucious/optimizer/interface/Constraint.h>

namespace lucious
{

namespace optimizer
{

Solver::Solver()
{

}

Solver::~Solver()
{
    for(auto constraint : _constraints)
    {
        delete constraint;
    }
}

void Solver::addConstraint(const Constraint& constraint)
{
    _constraints.push_back(constraint.clone());
}

}

}
