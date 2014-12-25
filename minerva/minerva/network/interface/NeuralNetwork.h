/* Author: Sudnya Padalikar
 * Date  : 08/09/2013
 * The interface of the Neural Network class
 */

#pragma once

// Standard Library Includes
#include <string>
#include <vector>
#include <map>
#include <random>

// Forward Declaration
namespace minerva { namespace network       { class Layer;                   } }
namespace minerva { namespace network       { class CostFunction;            } }
namespace minerva { namespace matrix        { class Matrix;                  } }
namespace minerva { namespace matrix        { class BlockSparseMatrix;       } }
namespace minerva { namespace matrix        { class BlockSparseMatrixVector; } }
namespace minerva { namespace optimizer     { class NeuralNetworkSolver;     } }

namespace minerva
{
namespace network
{

class NeuralNetwork
{
public:
    typedef matrix::Matrix                  Matrix;
    typedef matrix::BlockSparseMatrix       BlockSparseMatrix;
    typedef matrix::BlockSparseMatrixVector BlockSparseMatrixVector;
	typedef 

public:
    NeuralNetwork();

public:
	/*! \brief Initialize the network weights */
    void initializeRandomly(std::default_random_engine& engine, float epsilon = 3.0f);
	/*! \brief Initialize the network weights */
    void initializeRandomly(float epsilon = 3.0f);

public:
	/*! \brief Train the network on the specified input and reference. */
    void train(const Matrix& input, const Matrix& reference);
	/*! \brief Train the network on the specified input and reference. */
    void train(Matrix&& input, Matrix&& reference);
	/*! \brief Train the network on the specified input and reference. */
    void train(BlockSparseMatrix& input, BlockSparseMatrix& reference);

public:
	/*! \brief Get the cost and gradient. */
	float getCostAndGradient(BlockSparseMatrixVector& gradient, const BlockSparseMatrix& input, const BlockSparseMatrix& reference);
	/*! \brief Get the cost. */
	float getCost(const BlockSparseMatrix& input, const BlockSparseMatrix& reference);
	
	/*! \brief Get the cost and gradient. */
	float getCostAndGradient(Matrix& gradient, const Matrix& input, const Matrix& reference);
	/*! \brief Get the cost. */
	float getCost(const Matrix& input, const Matrix& reference);

public:
	/*! \brief Run input samples through the network, return the output */
    Matrix runInputs(const Matrix& m) const;
	/*! \brief Run input samples through the network, return the output */
    BlockSparseMatrix runInputs(const BlockSparseMatrix& m) const;

public:
	/*! \brief Add a new layer, the network takes ownership */
    void addLayer(Layer*);

public:
	/*! \brief Clear the network */
    void clear();

public:
	typedef std::unique_ptr<Layer>        LayerPointer;
	typedef std::unique_ptr<CostFunction> CostFunctionPointer;

	typedef std::unique_ptr<NeuralNetworkSolver> NeuralNetworkSolverPointer;

public:
          LayerPointer& operator[](size_t index);
    const LayerPointer& operator[](size_t index) const;

public:
          LayerPointer& back();
    const LayerPointer& back() const;

public:
          LayerPointer& front();
    const LayerPointer& front() const;

public:
    size_t size() const;
    bool   empty() const;

public:
    size_t getInputCount()  const;
    size_t getOutputCount() const;

    size_t getOutputCountForInputCount(size_t inputCount) const;

    size_t getInputBlockingFactor()  const;
    size_t getOutputBlockingFactor() const;

public:
    size_t totalNeurons()     const;
    size_t totalConnections() const;

public:
    size_t getFloatingPointOperationCount() const;

public:
    size_t totalWeights()     const;
    size_t totalActivations() const;

public:
    typedef std::vector<LayerPointer> LayerVector;

    typedef LayerVector::reverse_iterator	    reverse_iterator;
    typedef LayerVector::iterator	            iterator;
    typedef LayerVector::const_iterator         const_iterator;
    typedef LayerVector::const_reverse_iterator const_reverse_iterator;

public:
    iterator       begin();
    const_iterator begin() const;

    iterator       end();
    const_iterator end() const;

public:
    reverse_iterator       rbegin();
    const_reverse_iterator rbegin() const;

    reverse_iterator       rend();
    const_reverse_iterator rend() const;

public:
    NeuralNetwork getSubgraphConnectedToThisOutput(unsigned neuron) const;

public:
	/*! \brief Set the network cost function, the network takes ownership */
	void setCostFunction(CostFunction*);
	/*! \brief Get the network cost function, the network retains ownership */
	CostFunction* getCostFunction();

public:
	/*! \brief Set the network cost function, the network takes ownership */
	void setSolver(NeuralNetworkSolver*);
	/*! \brief Get the network cost function, the network retains ownership */
	NeuralNetworkSolver* getSolver();

public:
	/*! \brief Save the network to the tar file and header. */
	void save(util::TarArchive& archive) const;
	/*! \brief Intialize the network from the tar file and header. */
	void load(const util::TarArchive& archive, const std::string& name);

public:
    std::string shapeString() const;

public:
	NeuralNetwork(const NeuralNetwork& ) = delete;
	NeuralNetwork& operator=(const NeuralNetwork&) = delete;

private:
    BlockSparseMatrix convertToBlockSparseForLayerInput(const Layer& layer, const Matrix& m) const;
    BlockSparseMatrix convertToBlockSparseForLayerOutput(const Layer& layer, const Matrix& m) const;
    BlockSparseMatrix convertOutputToBlockSparse(const Matrix& m) const;
    void formatInputForLayer(const Layer& layer, BlockSparseMatrix& m) const;
    void formatOutputForLayer(const Layer& layer, BlockSparseMatrix& m) const;

private:
    LayerVector _layers;

private:
	CostFunctionPointer _costFunction;
	
private:
	NeuralNetworkSolverPointer _solver;


};

}

}

