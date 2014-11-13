/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The interface for the Layer class
 */

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix;       } }
namespace minerva { namespace matrix { class BlockSparseMatrixVector; } }

// Standard Library Includes
#include <random>
#include <set>

namespace minerva
{
namespace neuralnetwork
{

/* \brief A neural network layer interface. */
class Layer
{
public:
    typedef minerva::matrix::BlockSparseMatrix BlockSparseMatrix;
    typedef minerva::matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
    typedef std::set<size_t> NeuronSet;

public:
    virtual ~Layer();

public:
    virtual void initializeRandomly(std::default_random_engine& engine, float epsilon = 6.0f) = 0;
    virtual BlockSparseMatrix runForward(const BlockSparseMatrix& m) const = 0;
    virtual BlockSparseMatrix runReverse(BlockSparseMatrixVector& gradients, const BlockSparseMatrix& m) const = 0;

public:
    virtual       BlockSparseMatrixVector& weights()       = 0;
    virtual const BlockSparseMatrixVector& weights() const = 0;

public:
    virtual size_t getInputCount()  const = 0;
    virtual size_t getOutputCount() const = 0;

    virtual size_t getInputBlockingFactor()  const = 0;
    virtual size_t getOutputBlockingFactor() const = 0;

public:
    virtual size_t getOutputCountForInputCount(size_t inputCount) const = 0;

public:
    virtual size_t totalNeurons()	  const = 0;
    virtual size_t totalConnections() const = 0;

public:
    virtual size_t getFloatingPointOperationCount() const = 0;

public:
    virtual Layer* sliceSubgraphConnectedToTheseOutputs(
        const NeuronSet& outputs) const = 0;

};

}
}

