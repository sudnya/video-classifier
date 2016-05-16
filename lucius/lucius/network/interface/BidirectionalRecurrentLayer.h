/*  \file   BidirectionalRecurrentLayer.h
    \author Sudnya Diamos
    \date   May 9, 2016
    \brief  The interface for the BidirectionalRecurrentLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

/* \brief An implementation of a generic recurrent layer. */
class BidirectionalRecurrentLayer : public Layer
{
public:
    BidirectionalRecurrentLayer();
    BidirectionalRecurrentLayer(size_t size, size_t batchSize);
    BidirectionalRecurrentLayer(size_t size, size_t batchSize, const matrix::Precision&);
    virtual ~BidirectionalRecurrentLayer();

public:
    BidirectionalRecurrentLayer(const BidirectionalRecurrentLayer& );
    BidirectionalRecurrentLayer& operator=(const BidirectionalRecurrentLayer&);

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
    std::unique_ptr<MatrixVector> _parameters;

private:
    Matrix& _forwardWeights;
    Matrix& _bias;

private:
    Matrix& _recurrentForwardWeights;
    Matrix& _recurrentReverseWeights;

private:
    size_t _expectedBatchSize;

};
}

}




