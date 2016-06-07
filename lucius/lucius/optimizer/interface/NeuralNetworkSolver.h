/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the NeuralNetworkSolver class
 */

#pragma once

namespace lucius { namespace network { class NeuralNetwork; } }
namespace lucius { namespace network { class Bundle;        } }
namespace lucius { namespace matrix  { class Matrix;        } }

namespace lucius
{

namespace optimizer
{

/*! \brief A general purpose solver for a neural network */
class NeuralNetworkSolver
{
public:
    typedef network::NeuralNetwork NeuralNetwork;
    typedef matrix::Matrix         Matrix;
    typedef network::Bundle        Bundle;

public:
    NeuralNetworkSolver(NeuralNetwork* n);

    virtual ~NeuralNetworkSolver();

public:
    virtual double solve() = 0;

public:
    virtual NeuralNetworkSolver* clone() const = 0;

public:
    void setBundle(const Bundle* bundle);
    void setNetwork(NeuralNetwork* network);

public:
    static NeuralNetworkSolver* create(NeuralNetwork* n);

protected:
    NeuralNetwork* _network;

    const Bundle* _bundle;
};

}

}

