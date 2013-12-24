/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropagation.h>

namespace minerva
{
namespace optimizer
{
class Solver
{
    public:
        typedef minerva::neuralnetwork::BackPropagation BackPropagation;

    public:
        Solver(BackPropagation* d)
        : m_backPropDataPtr(d)
        {
        
        }
        
        virtual ~Solver()
        {
        
        }

    public:
        virtual void solve() = 0;

    public: 
        static Solver* create(BackPropagation* d);

    protected:
        BackPropagation* m_backPropDataPtr;
};

}
}
