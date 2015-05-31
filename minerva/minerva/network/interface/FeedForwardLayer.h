/*  \file   FeedForwardLayer.h
    \author Gregory Diamos
     \date   Dec 24, 2014
     \brief  The interface for the FeedForwardLayer class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/Layer.h>

// Standard Library Includes
#include <memory>

// Fordward Declarations
namespace minerva { namespace matrix { class Precision; } }

namespace minerva
{
namespace network
{

/* \brief An implementation of a generic fully connected feed forward layer. */
class FeedForwardLayer : public Layer
{
public:
    FeedForwardLayer();
    FeedForwardLayer(size_t inputs, size_t outputs);
    FeedForwardLayer(size_t inputs, size_t outputs, const matrix::Precision&);
    virtual ~FeedForwardLayer();

public:
    FeedForwardLayer(const FeedForwardLayer& );
    FeedForwardLayer& operator=(const FeedForwardLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(MatrixVector& activations) const;
    virtual Matrix runReverseImplementation(MatrixVector& gradients,
        MatrixVector& activations,
        const Matrix& deltas) const;

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

public:
    virtual void save(util::TarArchive& archive) const;
    virtual void load(const util::TarArchive& archive, const std::string& name);

public:
    virtual std::unique_ptr<Layer> clone() const;
    virtual std::unique_ptr<Layer> mirror() const;

public:
    virtual std::string getTypeName() const;

private:
    std::unique_ptr<MatrixVector> _parameters;

private:
    Matrix& _weights;
    Matrix& _bias;

};

}

}


