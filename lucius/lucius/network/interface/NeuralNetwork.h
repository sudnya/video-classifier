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
#include <memory>

// Forward Declaration
namespace lucius { namespace network   { class Layer;               } }
namespace lucius { namespace network   { class CostFunction;        } }
namespace lucius { namespace network   { class Bundle;              } }
namespace lucius { namespace matrix    { class Matrix;              } }
namespace lucius { namespace matrix    { class MatrixVector;        } }
namespace lucius { namespace matrix    { class Precision;           } }
namespace lucius { namespace matrix    { class Dimension;           } }
namespace lucius { namespace util      { class InputTarArchive;     } }
namespace lucius { namespace util      { class OutputTarArchive;    } }
namespace lucius { namespace util      { class PropertyTree;        } }
namespace lucius { namespace optimizer { class NeuralNetworkSolver; } }

namespace lucius
{
namespace network
{

class NeuralNetwork
{
public:
    typedef matrix::Matrix                 Matrix;
    typedef matrix::MatrixVector           MatrixVector;
    typedef matrix::Dimension              Dimension;
    typedef optimizer::NeuralNetworkSolver NeuralNetworkSolver;

public:
    NeuralNetwork();
    ~NeuralNetwork();

public:
    /*! \brief Initialize the network weights */
    void initialize();

public:
    /*! \brief Get the cost and gradient. */
    Bundle getCostAndGradient(const Bundle& bundle);
    /*! \brief Get the cost. */
    Bundle getCost(const Bundle& bundle);

public:
    /*! \brief Get the cost and gradient with respect to the inputs. */
    Bundle getInputCostAndGradient(const Bundle& bundle);

public:
    /*! \brief Run input samples through the network, return the output */
    Matrix runInputs(const Matrix& input);

    /*! \brief Run input samples through the network, update the bundle. */
    Bundle runInputs(const Bundle& bundle);

public:
    /*! \brief Add an existing layer, the network takes ownership */
    void addLayer(std::unique_ptr<Layer>&& );

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
    const matrix::Precision& precision() const;

public:
    Dimension getInputSize()  const;
    Dimension getOutputSize() const;

public:
    size_t getInputCount()  const;
    size_t getOutputCount() const;

public:
    size_t totalNeurons()     const;
    size_t totalConnections() const;

public:
    size_t getFloatingPointOperationCount() const;
    size_t getActivationMemory() const;
    size_t getParameterMemory() const;

public:
    /*! \brief Train the network on the specified input and reference, return the cost. */
    double train(const Matrix& input, const Matrix& referenceOutput);

    /*! \brief Train the network on the input and output activations, update the bundle. */
    Bundle train(const Bundle& bundle);

public:
    typedef std::vector<LayerPointer> LayerVector;

    typedef LayerVector::reverse_iterator       reverse_iterator;
    typedef LayerVector::iterator               iterator;
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
    /*! \brief Set the network cost function, the network takes ownership */
    void setCostFunction(CostFunction*);
    /*! \brief Get the network cost function, the network retains ownership */
    CostFunction* getCostFunction();
    /*! \brief Get the network cost function, the network retains ownership */
    const CostFunction* getCostFunction() const;

public:
    /*! \brief Set the network cost function, the network takes ownership */
    void setSolver(NeuralNetworkSolver*);
    /*! \brief Get the network cost function, the network retains ownership */
    NeuralNetworkSolver* getSolver();
    /*! \brief Get the network cost function, the network retains ownership */
    const NeuralNetworkSolver* getSolver() const;

public:
    /*! \brief Save the network to the tar file and header. */
    void save(util::OutputTarArchive& archive, util::PropertyTree& properties) const;
    /*! \brief Intialize the network from the tar file and header. */
    void load(util::InputTarArchive& archive, const util::PropertyTree& properties);

public:
    /*! \brief Indicate that the network is being trained or not. */
    void setIsTraining(bool training);

public:
    std::string shapeString() const;

public:
    NeuralNetwork(const NeuralNetwork& );
    NeuralNetwork& operator=(const NeuralNetwork&);

private:
    LayerVector _layers;

private:
    CostFunctionPointer _costFunction;

private:
    NeuralNetworkSolverPointer _solver;


};

}

}

