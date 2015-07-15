/*! \file   Solver.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Saturday March 8, 2014
    \brief  The source file for the Solver class.
*/

// Lucius Includes
#include <lucius/optimizer/interface/Solver.h>
#include <lucius/optimizer/interface/Constraint.h>

namespace lucius
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
