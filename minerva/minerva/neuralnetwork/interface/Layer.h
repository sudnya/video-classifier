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
    virtual void initializeRandomly(std::default_random_engine& engine,
		float epsilon = 6.0f) = 0;

public:
	virtual BlockSparseMatrix train(const BlockSparseMatrix& m) const = 0;

public:
    virtual BlockSparseMatrix runForward(const BlockSparseMatrix& m) const = 0;
    virtual BlockSparseMatrix runReverse(BlockSparseMatrixVector& gradients,
		const BlockSparseMatrix& m) const = 0;

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

public:
	/*! \brief Save the layer to the tar file and header. */
	virtual void save(util::TarArchive& archive) const = 0;
	/*! \brief Intialize the layer from the tar file and header. */
	virtual void load(const util::TarArchive& archive) = 0;

public:
	/*! \brief Set the activation function, the layer takes ownership. */
	void setActivationFunction(ActivationFunction*);
	/*! \brief Get the activation function, the layer retains ownership. */
	ActivationFunction* getActivationFunction();

public:
	/*! \brief Set the activation cost function component, the layer takes ownership. */
	void setActivationCostFunction(ActivationCostFunction*);
	/*! \brief Get the activation cost function component, the layer retains ownership. */
	ActivationCostFunction* getActivationCostFunction();
	
public:
	/*! \brief Set the weight cost function component, the layer takes ownership. */
	virtual void setWeightCostFunction(WeightCostFunction*);
	/*! \brief Get the weight cost function component, the layer retains ownership. */
	virtual void WeightCostFunction* getWeightCostFunction();

private:
	std::unique_ptr<ActivationFunction>     _activationFunction;
	std::unique_ptr<ActivationCostFunction> _activationCostFunction;
	std::unique_ptr<WeightCostFunction>     _weightCostFunction;

};

}
}

