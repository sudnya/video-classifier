/* Author: Sudnya Padalikar
 * Date  : 08/11/2013
 * The interface for the Layer class
 */

#pragma once

// Standard Library Includes
#include <random>
#include <set>
#include <memory>

// Forward Declarations
namespace lucious { namespace matrix  { class Matrix;                 } }
namespace lucious { namespace matrix  { class Dimension;              } }
namespace lucious { namespace matrix  { class MatrixVector;           } }
namespace lucious { namespace matrix  { class Precision;              } }
namespace lucious { namespace network { class ActivationFunction;     } }
namespace lucious { namespace network { class ActivationCostFunction; } }
namespace lucious { namespace network { class WeightCostFunction;     } }
namespace lucious { namespace util    { class TarArchive;             } }

namespace lucious
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

public:
    Layer();
    virtual ~Layer();

public:
    Layer(const Layer& );
    Layer& operator=(const Layer&);

public:
    virtual void initialize() = 0;

public:
    void runForward(MatrixVector& activations) const;
    Matrix runReverse(MatrixVector& gradients,
        MatrixVector& activations,
        const Matrix& deltas) const;

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
    virtual size_t totalNeurons()      const = 0;
    virtual size_t totalConnections() const = 0;

public:
    virtual size_t getFloatingPointOperationCount() const = 0;

public:
    /*! \brief Save the layer to the tar file and header. */
    virtual void save(util::TarArchive& archive) const = 0;
    /*! \brief Intialize the layer from the tar file and header. */
    virtual void load(const util::TarArchive& archive, const std::string& name) = 0;

public:
    virtual std::unique_ptr<Layer> clone() const = 0;
    virtual std::unique_ptr<Layer> mirror() const = 0;

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

protected:
    virtual void runForwardImplementation(MatrixVector& m) const = 0;
    virtual Matrix runReverseImplementation(MatrixVector& gradients,
        MatrixVector& activations, const Matrix& deltas) const = 0;

private:
    std::unique_ptr<ActivationFunction>     _activationFunction;
    std::unique_ptr<ActivationCostFunction> _activationCostFunction;
    std::unique_ptr<WeightCostFunction>     _weightCostFunction;

};

}

}

