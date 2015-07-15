/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the Solver class
 */


#include <lucius/optimizer/interface/NeuralNetworkSolver.h>
#include <lucius/optimizer/interface/SimpleNeuralNetworkSolver.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

namespace lucius
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

