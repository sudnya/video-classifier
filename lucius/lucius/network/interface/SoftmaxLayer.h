/*  \file   SoftmaxLayer.h
    \author Gregory Diamos
    \date   Dec 24, 2014
    \brief  The interface for the SoftmaxLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

/* \brief An implementation of a softmax layer. */
class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer();
    SoftmaxLayer(const Dimension& inputSize);
    SoftmaxLayer(const Dimension& inputSize, const matrix::Precision&);
    virtual ~SoftmaxLayer();

public:
    SoftmaxLayer(const SoftmaxLayer& );
    SoftmaxLayer& operator=(const SoftmaxLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(Bundle& bundle);
    virtual void runReverseImplementation(Bundle& bundle);

public:
    virtual       MatrixVector& weights();
    virtual const MatrixVector& weights() const;

public:
    virtual const matrix::Precision& precision() const;

public:
    virtual double computeWeightCost() const;

public:
    virtual Dimension getInputSize()  const;
    virtual Dimension getOutputSize() const;

public:
    virtual size_t getInputCount()  const;
    virtual size_t getOutputCount() const;

public:
    virtual size_t totalNeurons()      const;
    virtual size_t totalConnections() const;

public:
    virtual size_t getFloatingPointOperationCount() const;
    virtual size_t getActivationMemory() const;

public:
    virtual void save(util::OutputTarArchive& archive, util::PropertyTree& properties) const;
    virtual void load(util::InputTarArchive& archive, const util::PropertyTree& properties);

public:
    virtual std::unique_ptr<Layer> clone() const;

public:
    virtual std::string getTypeName() const;

private:
    std::unique_ptr<matrix::Dimension> _inputSize;
    std::unique_ptr<matrix::Precision> _precision;

};

}

}



