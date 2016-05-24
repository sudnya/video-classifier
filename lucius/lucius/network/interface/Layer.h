/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The interface for the Layer class
 */

#pragma once

// Standard Library Includes
#include <random>
#include <set>
#include <map>
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix  { class Matrix;                 } }
namespace lucius { namespace matrix  { class Dimension;              } }
namespace lucius { namespace matrix  { class MatrixVector;           } }
namespace lucius { namespace matrix  { class Precision;              } }
namespace lucius { namespace network { class ActivationFunction;     } }
namespace lucius { namespace network { class ActivationCostFunction; } }
namespace lucius { namespace network { class WeightCostFunction;     } }
namespace lucius { namespace network { class Bundle;                 } }
namespace lucius { namespace util    { class InputTarArchive;        } }
namespace lucius { namespace util    { class OutputTarArchive;       } }
namespace lucius { namespace util    { class PropertyTree;           } }

namespace lucius
{
namespace network
{

/* \brief A neural network layer interface. */
class Layer
{
public:
    typedef matrix::Matrix       Matrix;
    typedef matrix::MatrixVector MatrixVector;
    typedef matrix::Dimension    Dimension;
    typedef network::Bundle      Bundle;

public:
    Layer();
    virtual ~Layer();

public:
    Layer(const Layer& );
    Layer& operator=(const Layer&);

public:
    virtual void initialize() = 0;

public:
    void runForward(Bundle& bundle);
    void runReverse(Bundle& bundle);

public:
    virtual void popReversePropagationData();
    virtual void clearReversePropagationData();

public:
    virtual       MatrixVector& weights()       = 0;
    virtual const MatrixVector& weights() const = 0;

public:
    virtual const matrix::Precision& precision() const = 0;

public:
    virtual double computeWeightCost() const = 0;

public:
    virtual Dimension getInputSize()  const = 0;
    virtual Dimension getOutputSize() const = 0;

public:
    virtual size_t getInputCount()  const = 0;
    virtual size_t getOutputCount() const = 0;

public:
    virtual size_t totalNeurons()     const = 0;
    virtual size_t totalConnections() const = 0;

public:
    virtual size_t getFloatingPointOperationCount() const = 0;
    virtual size_t getActivationMemory() const = 0;
    virtual size_t getParameterMemory()  const;

public:
    /*! \brief Save the layer to the tar file and header. */
    virtual void save(util::OutputTarArchive& archive, util::PropertyTree& properties) const = 0;
    /*! \brief Intialize the layer from the tar file and header. */
    virtual void load(util::InputTarArchive& archive, const util::PropertyTree& properties) = 0;

public:
    virtual std::unique_ptr<Layer> clone() const = 0;

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
    /*! \brief Indicate that the layer is being trained or not. */
    void setIsTraining(bool training);

    /*! \brief Query whether or not the network is being trained. */
    bool getIsTraining() const;

public:
    /*! \brief Query whether or not the layer supports multiple inputs and outputs. */
    bool getSupportsMultipleInputsAndOutputs() const;

public:
    /*! \brief Indicate whether or not the layer should compute deltas. */
    virtual void setShouldComputeDeltas(bool shouldComputeDeltas);

    /*! \brief Query whether or not the layer should compute deltas. */
    bool getShouldComputeDeltas() const;

public:
    std::string shapeString() const;
    std::string resourceString() const;

protected:
    virtual void runForwardImplementation(Bundle& bundle) = 0;
    virtual void runReverseImplementation(Bundle& bundle) = 0;

protected:
    /*! \brief Save the layer to the tar file and header. */
    virtual void saveLayer(util::OutputTarArchive& archive, util::PropertyTree& properties) const;
    /*! \brief Intialize the layer from the tar file and header. */
    virtual void loadLayer(util::InputTarArchive& archive, const util::PropertyTree& properties);

protected:
    /*! \brief Cache a matrix for back propagation */
    void saveMatrix(const std::string& name, const Matrix& data);
    /*! \brief Load a cached matrix. */
    Matrix loadMatrix(const std::string& name);

protected:
    /*! \brief Indicate that the layer supports multiple inputs and outputs. */
    void setSupportsMultipleInputsAndOutputs(bool supportMultipleIOs);

private:
    std::unique_ptr<ActivationFunction>     _activationFunction;
    std::unique_ptr<ActivationCostFunction> _activationCostFunction;
    std::unique_ptr<WeightCostFunction>     _weightCostFunction;

private:
    std::map<std::string, MatrixVector> _matrixCache;

private:
    bool _isTraining;
    bool _shouldComputeDeltas;
    bool _supportsMultipleInputsAndOutputs;

};

}

}

