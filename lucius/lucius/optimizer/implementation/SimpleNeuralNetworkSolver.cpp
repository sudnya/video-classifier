/*    \file   SimpleNeuralNetworkSolver.cpp
    \date   Sunday December 26, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the SimpleNeuralNetworkSolver class.
*/

// Lucius Includes
#include <lucius/optimizer/interface/SimpleNeuralNetworkSolver.h>
#include <lucius/optimizer/interface/GeneralDifferentiableSolver.h>
#include <lucius/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <lucius/optimizer/interface/CostAndGradientFunction.h>

#include <lucius/network/interface/NeuralNetwork.h>
#include <lucius/network/interface/Layer.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/SystemCompatibility.h>
#include <lucius/util/interface/debug.h>

// Standard Libary Includes
#include <list>
#include <set>
#include <stack>

namespace lucius
{

namespace optimizer
{

typedef network::NeuralNetwork                    NeuralNetwork;
typedef matrix::Matrix                            Matrix;
typedef GeneralDifferentiableSolver::MatrixVector MatrixVector;

SimpleNeuralNetworkSolver::SimpleNeuralNetworkSolver(NeuralNetwork* n)
: NeuralNetworkSolver(n), _solver(GeneralDifferentiableSolverFactory::create())
{

}

SimpleNeuralNetworkSolver::~SimpleNeuralNetworkSolver()
{

}

SimpleNeuralNetworkSolver::SimpleNeuralNetworkSolver(const SimpleNeuralNetworkSolver& s)
: NeuralNetworkSolver(s), _solver(GeneralDifferentiableSolverFactory::create())
{

}

SimpleNeuralNetworkSolver& SimpleNeuralNetworkSolver::operator=(const SimpleNeuralNetworkSolver& s)
{
    if(&s == this)
    {
        return *this;
    }

    NeuralNetworkSolver::operator=(s);

    return *this;
}

static void setWeights(NeuralNetwork& network, const MatrixVector& weights)
{
    size_t weight = 0;

    for(auto& layer : network)
    {
        for(auto& weightMatrix : layer->weights())
        {
            weightMatrix = weights[weight++];
        }
    }
}

class NeuralNetworkCostAndGradient : public CostAndGradientFunction
{
public:
    NeuralNetworkCostAndGradient(NeuralNetwork* n, const Matrix* i, const Matrix* r)
    : _network(n), _input(i), _reference(r)
    {

    }

    virtual ~NeuralNetworkCostAndGradient()
    {

    }

public:
    virtual double computeCostAndGradient(MatrixVector& gradient,
        const MatrixVector& weights) const
    {
        setWeights(*_network, weights);

        double newCost = _network->getCostAndGradient(gradient, *_input, *_reference);

        if(util::isLogEnabled("SimpleNeuralNetworkSolver::Detail"))
        {
            util::log("SimpleNeuralNetworkSolver::Detail") << " new gradient is : " << gradient[1].toString();
        }

        util::log("SimpleNeuralNetworkSolver::Detail") << " new cost is : " << newCost << "\n";

        return newCost;
    }

private:
    NeuralNetwork* _network;
    const Matrix*  _input;
    const Matrix*  _reference;
};

static MatrixVector getWeights(NeuralNetwork* network)
{
    MatrixVector weights;

    for(auto& layer : *network)
    {
        weights.push_back(layer->weights());
    }

    return weights;
}

static double differentiableSolver(NeuralNetwork* network, const Matrix* input, const Matrix* reference, GeneralDifferentiableSolver* solver)
{
    util::log("SimpleNeuralNetworkSolver") << "  starting general solver\n";
    double newCost = std::numeric_limits<double>::infinity();

    if(!solver)
    {
        util::log("SimpleNeuralNetworkSolver") << "   failed to allocate solver\n";
        return newCost;
    }

    NeuralNetworkCostAndGradient costAndGradient(network, input, reference);

    auto weights = getWeights(network);

    newCost = solver->solve(weights, costAndGradient);

    util::log("SimpleNeuralNetworkSolver") << "   solver produced new cost: "
        << newCost << ".\n";

    setWeights(*network, weights);

    return newCost;
}


void SimpleNeuralNetworkSolver::solve()
{
    util::log("SimpleNeuralNetworkSolver") << "Solve\n";
    util::log("SimpleNeuralNetworkSolver")
        << " no need for tiling, solving entire network at once.\n";
    differentiableSolver(_network, _input, _reference, _solver.get());
}

NeuralNetworkSolver* SimpleNeuralNetworkSolver::clone() const
{
    return new SimpleNeuralNetworkSolver(*this);
}

}

}



