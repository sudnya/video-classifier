/*  \file   RecurrentLayer.h
    \author Gregory Diamos
     \date   Dec 24, 2014
     \brief  The interface for the RecurrentLayer class.
*/

#pragma once

// Lucious Includes
#include <lucious/network/interface/Layer.h>

namespace lucious
{
namespace network
{

/* \brief An implementation of a generic recurrent layer. */
class RecurrentLayer : public Layer
{
public:
    RecurrentLayer();
    RecurrentLayer(size_t size, size_t batchSize);
    RecurrentLayer(size_t size, size_t batchSize, const matrix::Precision&);
    virtual ~RecurrentLayer();

public:
    RecurrentLayer(const RecurrentLayer& );
    RecurrentLayer& operator=(const RecurrentLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(MatrixVector& activations) const;
    virtual Matrix runReverseImplementation(MatrixVector& gradients,
        MatrixVector& activations, const Matrix& deltas) const;

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
    Matrix& _forwardWeights;
    Matrix& _bias;

private:
    Matrix& _recurrentWeights;

private:
    size_t _expectedBatchSize;

};
}

}



