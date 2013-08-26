/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Gradient descent Solver class 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropData.h>

namespace minerva
{
namespace optimizer
{
class GradientDescentSolver : public Solver
{
    public:
        typedef minerva::neuralnetwork::BackPropData BackPropData;
    public:
        GradientDescentSolver(BackPropData* d) : Solver(d)
    {
    }
        void solve();


};

}
}
