/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class 
 */


#include <minerva/optimizer/interface/Solver.h>
#include <minerva/optimizer/interface/MultiLevelOptimizer.h>
#include <minerva/optimizer/interface/TiledConvolutionalSolver.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

namespace minerva
{
namespace optimizer
{

void Solver::solve()
{
    assertM(false, "Not implemented yet");
}

Solver* Solver::create(BackPropagation* d)
{
	auto solverName = util::KnobDatabase::getKnobValue("Solver::Type",
		"TiledConvolutionalSolver");
		//"MultiLevelOptimizer");	

	if(solverName == "MultiLevelOptimizer")
	{
		return new MultiLevelOptimizer(d);
	}

	// Fall back to tiled solver
	return new TiledConvolutionalSolver(d);
}

}
}

