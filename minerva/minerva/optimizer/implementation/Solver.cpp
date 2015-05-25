/*! \file   Solver.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Saturday March 8, 2014
    \brief  The source file for the Solver class.
*/

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/Constraint.h>

namespace minerva
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
