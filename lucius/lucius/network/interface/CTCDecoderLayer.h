/*  \file   CTCDecoderLayer.h
    \author Gregory Diamos
    \date   October 9, 2016
    \brief  The interface for the CTCDecoderLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

class CTCDecoderLayerImplementation;

/*! \brief A class for decoding the output of a CTC network.

    Note: we are assuming the separator character is the first (0th) character.
*/
class CTCDecoderLayer : public Layer
{
public:
    CTCDecoderLayer();
    CTCDecoderLayer(const Dimension& inputSize);
    CTCDecoderLayer(const Dimension& inputSize, size_t beamSize);
    CTCDecoderLayer(const Dimension& inputSize, size_t beamSize,
        const std::string& costFunctionName, double costFunctionWeight,
        const matrix::Precision&);
    virtual ~CTCDecoderLayer();

public:
    CTCDecoderLayer(const CTCDecoderLayer& layer);
    CTCDecoderLayer& operator=(const CTCDecoderLayer& layer);

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
    virtual void save(util::OutputTarArchive& archive,
        util::PropertyTree& properties) const;
    virtual void load(util::InputTarArchive& archive,
        const util::PropertyTree& properties);

public:
    virtual std::unique_ptr<Layer> clone() const;

public:
    virtual std::string getTypeName() const;

private:
    std::unique_ptr<CTCDecoderLayerImplementation> _implementation;

};

}
}

