/*  \file   SparseSearchLayer.h
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SparseSearchLayer class.
*/

#pragma once

// Lucius Includes
#include <lucius/network/interface/Layer.h>

namespace lucius
{
namespace network
{

/* \brief A layer that searches over an input in response to a command, generating an output. */
class SparseSearchLayer : public Layer
{
public:
    SparseSearchLayer();
    SparseSearchLayer(size_t size);
    SparseSearchLayer(size_t size, const matrix::Precision&);
    virtual ~SparseSearchLayer();

public:
    SparseSearchLayer(const SparseSearchLayer& );
    SparseSearchLayer& operator=(const SparseSearchLayer&);

public:
    virtual void initialize();

public:
    virtual void runForwardImplementation(MatrixVector& outputActivations,
        const MatrixVector& inputActivations);
    virtual Matrix runReverseImplementation(MatrixVector& gradients,
        MatrixVector& inputDeltas,
        const Matrix& outputDeltas);

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
    std::unique_ptr<MatrixVector> _parameters;

private:
    Matrix& _forwardWeights;
    Matrix& _bias;

private:
    Matrix& _recurrentWeights;

};
}

}





