/*  \file   DropoutLayer.h
    \author Gregory Diamos
    \date   February 21, 2017
    \brief  The interface for the DropoutLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

/* \brief A layer that performs dropout on the input. */
class DropoutLayer: public Layer
{
public:
    DropoutLayer();
    DropoutLayer(const Dimension& inputSize, double dropoutRatio);
    DropoutLayer(const Dimension& inputSize, double dropoutRatio, const matrix::Precision&);
    virtual ~DropoutLayer();

public:
    DropoutLayer(const DropoutLayer& );
    DropoutLayer& operator=(const DropoutLayer&);

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

private:
    size_t _trainingIteration;

private:
    double _dropoutRatio;

};

}

}





