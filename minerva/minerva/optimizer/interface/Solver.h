/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class 
 */

#pragma once

#include <minerva/neuralnetwork/interface/BackPropData.h>

namespace minerva
{
namespace optimizer
{
class Solver
{
    public:
        typedef minerva::neuralnetwork::BackPropData BackPropData;

    public:
        Solver(BackPropData* d)
        : m_backPropDataPtr(d)
        {
        
        }
        
        virtual ~Solver()
        {
        
        }

    public:
        virtual void solve() = 0;

    public: 
        static Solver* create(BackPropData* d);

    protected:
        BackPropData* m_backPropDataPtr;
};

}
}
