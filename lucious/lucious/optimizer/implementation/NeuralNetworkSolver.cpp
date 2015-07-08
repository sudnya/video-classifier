/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class
 */


#include <lucious/optimizer/interface/NeuralNetworkSolver.h>
#include <lucious/optimizer/interface/SimpleNeuralNetworkSolver.h>

#include <lucious/util/interface/debug.h>
#include <lucious/util/interface/Knobs.h>

namespace lucious
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

void NeuralNetworkSolver::setNetwork(NeuralNetwork* network)
{
    _network = network;
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

