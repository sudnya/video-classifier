/*  \file   MaxPoolingLayer.h
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the MaxPoolingLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

// Standard Library Includes
#include <memory>

namespace lucius
{
namespace network
{

/* \brief An implementation of max pooling. */
class MaxPoolingLayer : public Layer
{
public:
    MaxPoolingLayer();
    MaxPoolingLayer(const Dimension& size);
    MaxPoolingLayer(const Dimension& size, const matrix::Precision&);
    virtual ~MaxPoolingLayer();

public:
    MaxPoolingLayer(const MaxPoolingLayer& );
    MaxPoolingLayer& operator=(const MaxPoolingLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(MatrixVector& activations);
    virtual Matrix runReverseImplementation(MatrixVector& gradients,
        MatrixVector& activations,
        const Matrix& deltas);

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
    virtual std::unique_ptr<Layer> mirror() const;

public:
    virtual std::string getTypeName() const;

private:
    std::unique_ptr<matrix::Dimension> _filterSize;

};

}

}




