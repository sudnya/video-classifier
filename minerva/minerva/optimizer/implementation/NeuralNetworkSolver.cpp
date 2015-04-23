/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class
 */


#include <minerva/optimizer/interface/NeuralNetworkSolver.h>
#include <minerva/optimizer/interface/SimpleNeuralNetworkSolver.h>

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

void NeuralNetworkSolver::setInput(const Matrix* input)
{
    _input = input;
}

void NeuralNetworkSolver::setReference(const Matrix* reference)
{
    _reference = reference;
}

NeuralNetworkSolver* NeuralNetworkSolver::create(NeuralNetwork* n)
{
    auto solverName = util::KnobDatabase::getKnobValue("Solver::Type",
        "SimpleNeuralNetworkSolver");

    // Fall back to simple solver
    return new SimpleNeuralNetworkSolver(n);
}

}
}

