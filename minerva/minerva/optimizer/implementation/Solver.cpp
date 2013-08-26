/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class 
 */


#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/GradientDescentSolver.h>
#include <minerva/util/interface/debug.h>

namespace minerva
{
namespace optimizer
{

void Solver::solve()
{
    assertM(false, "Not implemented yet");
}

Solver* Solver::create(BackPropData* d)
{
    return new GradientDescentSolver(d);
}

}
}

