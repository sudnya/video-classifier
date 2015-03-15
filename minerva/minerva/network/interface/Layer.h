/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The interface for the Layer class
 */

#pragma once

// Forward Declarations
namespace minerva { namespace matrix  { class Matrix;                  } }
namespace minerva { namespace network { class ActivationFunction;      } }
namespace minerva { namespace network { class ActivationCostFunction;  } }
namespace minerva { namespace network { class WeightCostFunction;      } }
namespace minerva { namespace matrix  { class SparseMatrixFormat;      } }
namespace minerva { namespace util    { class TarArchive;              } }

// Standard Library Includes
#include <random>
#include <set>

namespace minerva
{
namespace network
{

/* \brief A neural network layer interface. */
class Layer
{
public:
    typedef matrix::Matrix                  Matrix;
    typedef matrix::MatrixVector            MatrixVector;
	typedef matrix::SparseMatrixFormat      SparseMatrixFormat;
	typedef std::vector<SparseMatrixFormat> SparseMatrixVectorFormat;
    typedef std::set<size_t>                NeuronSet;

public:
	Layer();
    virtual ~Layer();

public:
	Layer(const Layer& );
	Layer& operator=(const Layer&);

public:
    virtual void initializeRandomly(std::default_random_engine& engine,
		float epsilon = 6.0f) = 0;

public:
    virtual Matrix runForward(const Matrix& m) const = 0;
    virtual Matrix runReverse(MatrixVector& gradients,
		const Matrix& inputActivations,
		const Matrix& outputActivations,
		const Matrix& deltas) const = 0;

public:
    virtual       MatrixVector& weights()       = 0;
    virtual const MatrixVector& weights() const = 0;

public:
	virtual float computeWeightCost() const = 0;

public:
    virtual size_t getInputCount()  const = 0;
    virtual size_t getOutputCount() const = 0;

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
	virtual void load(const util::TarArchive& archive, const std::string& name) = 0;

public:
	/*! \brief Move the weight matrices outside of the network. */
	virtual void extractWeights(MatrixVector&  weights) = 0;
	/*! \brief Replace the weight matrices contained in the network with the specified weights */
	virtual void restoreWeights(MatrixVector&& weights) = 0;
	/*! \brief Get the sparse matrix format used by the weight matrices */
	virtual SparseMatrixVectorFormat getWeightFormat() const = 0;

public:
	virtual Layer* clone() const = 0;
	virtual Layer* mirror() const = 0;

public:
	virtual std::string getTypeName() const = 0;

public:
	/*! \brief Set the activation function, the layer takes ownership. */
	void setActivationFunction(ActivationFunction*);
	/*! \brief Get the activation function, the layer retains ownership. */
	ActivationFunction* getActivationFunction();
	/*! \brief Get the activation function, the layer retains ownership. */
	const ActivationFunction* getActivationFunction() const;

public:
	/*! \brief Set the activation cost function component, the layer takes ownership. */
	void setActivationCostFunction(ActivationCostFunction*);
	/*! \brief Get the activation cost function component, the layer retains ownership. */
	ActivationCostFunction* getActivationCostFunction();
	/*! \brief Get the activation cost function component, the layer retains ownership. */
	const ActivationCostFunction* getActivationCostFunction() const;
	
public:
	/*! \brief Set the weight cost function component, the layer takes ownership. */
	void setWeightCostFunction(WeightCostFunction*);
	/*! \brief Get the weight cost function component, the layer retains ownership. */
	WeightCostFunction* getWeightCostFunction();
	/*! \brief Get the weight cost function component, the layer retains ownership. */
	const WeightCostFunction* getWeightCostFunction() const;

public:
	std::string shapeString() const;

private:
	std::unique_ptr<ActivationFunction>     _activationFunction;
	std::unique_ptr<ActivationCostFunction> _activationCostFunction;
	std::unique_ptr<WeightCostFunction>     _weightCostFunction;

};

}

}

