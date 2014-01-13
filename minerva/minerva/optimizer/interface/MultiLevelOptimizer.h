/* Author: Sudnya Padalikar
 * Date  : 09/01/2013
 * The interface for the MultiLevelOptimizer class
 * We use the following heuristics in this optimizer:
 * greedy, local search, tabu search, simulated annealing 
 */

#pragma once

// Minerva Includes
#include <minerva/optimizer/interface/Solver.h>

// Standard Library Includes
#include <random>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{
namespace optimizer
{
class MultiLevelOptimizer : public Solver
{
    public:
        typedef minerva::matrix::Matrix Matrix;
        
    public:
        MultiLevelOptimizer(BackPropagation* d);
        void solve();
    
    private:
        void greedy();
        void addToTabuList();

	private:
		std::default_random_engine generator;

};

}
}

