/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class 
 */


#include <minerva/optimizer/interface/NeuralNetworkSolver.h>
#include <minerva/optimizer/interface/TiledConvolutionalSolver.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Knobs.h>

namespace minerva
{
namespace optimizer
{

NeuralNetworkSolver::NeuralNetworkSolver(NeuralNetwork* n)
: _network(n)
{

}

NeuralNetworkSolver::~NeuralNetworkSolver()
{

}

void NeuralNetworkSolver::solve()
{
    assertM(false, "Not implemented yet");
}

void NeuralNetworkSolver::setInput(const BlockSparseMatrix* input)
{
	_input = input;
}

void NeuralNetworkSolver::setReference(const BlockSparseMatrix* reference)
{
	_reference = reference;
}

NeuralNetworkSolver* NeuralNetworkSolver::create(NeuralNetwork* n)
{
	auto solverName = util::KnobDatabase::getKnobValue("Solver::Type",
		"TiledConvolutionalSolver");

	// Fall back to tiled solver
	return new TiledConvolutionalSolver(n);
}

}
}

