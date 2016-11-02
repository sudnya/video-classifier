/*  \file   AudioConvolutionalLayer.h
    \author Gregory Diamos
    \date   Dec 24, 2014
    \brief  The interface for the AudioConvolutionalLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

// Forward Declarations
namespace lucius { namespace network { class ConvolutionalLayer; } }

namespace lucius
{
namespace network
{

/* \brief An implementation of a generic recurrent layer. */
class AudioConvolutionalLayer : public Layer
{
public:
    AudioConvolutionalLayer();
    virtual ~AudioConvolutionalLayer();

public:
    AudioConvolutionalLayer(const matrix::Dimension& inputSize,
        const matrix::Dimension& filterSize,
        const matrix::Dimension& filterStride, const matrix::Dimension& inputPadding);
    AudioConvolutionalLayer(const matrix::Dimension& inputSize,
        const matrix::Dimension& filterSize,
        const matrix::Dimension& filterStride, const matrix::Dimension& inputPadding,
        const matrix::Precision&);

public:
    AudioConvolutionalLayer(const AudioConvolutionalLayer& );
    AudioConvolutionalLayer& operator=(const AudioConvolutionalLayer&);

public:
    virtual void initialize();

public:
    virtual void popReversePropagationData();
    virtual void clearReversePropagationData();

public:
    virtual void setShouldComputeDeltas(bool shouldComputeDeltas);

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
    std::unique_ptr<ConvolutionalLayer> _layer;

};

}

}





