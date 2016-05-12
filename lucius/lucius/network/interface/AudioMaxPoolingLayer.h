/*  \file   AudioMaxPoolingLayer.h
    \author Gregory Diamos
    \date   Dec 24, 2014
    \brief  The interface for the AudioMaxPoolingLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

// Forward Declarations
namespace lucius { namespace network { class MaxPoolingLayer; } }

namespace lucius
{
namespace network
{

/* \brief An implementation of a generic recurrent layer. */
class AudioMaxPoolingLayer : public Layer
{
public:
    AudioMaxPoolingLayer();
    virtual ~AudioMaxPoolingLayer();

public:
    AudioMaxPoolingLayer(const Dimension& inputSize, const Dimension& filterSize);
    AudioMaxPoolingLayer(const Dimension& inputSize, const Dimension& filterSize,
        const matrix::Precision&);

public:
    AudioMaxPoolingLayer(const AudioMaxPoolingLayer& );
    AudioMaxPoolingLayer& operator=(const AudioMaxPoolingLayer&);

public:
    virtual void initialize();

public:
    virtual void popReversePropagationData();
    virtual void clearReversePropagationData();

public:
    virtual void setShouldComputeDeltas(bool shouldComputeDeltas);

public:
    virtual void runForwardImplementation(MatrixVector& outputActivations,
        const MatrixVector& inputActivations);
    virtual void runReverseImplementation(MatrixVector& gradients,
        MatrixVector& inputDeltas, const MatrixVector& outputDeltas);

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
    std::unique_ptr<MaxPoolingLayer> _layer;

};

}

}






