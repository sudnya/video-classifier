/*  \file   BatchNormalizationLayer.h
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the BatchNormalizationLayer class.
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

/* \brief An implementation of batch normalization. */
class BatchNormalizationLayer : public Layer
{
public:
    BatchNormalizationLayer();
    BatchNormalizationLayer(size_t inputs);
    BatchNormalizationLayer(const Dimension& size);
    BatchNormalizationLayer(const Dimension& size, const matrix::Precision&);
    virtual ~BatchNormalizationLayer();

public:
    BatchNormalizationLayer(const BatchNormalizationLayer& );
    BatchNormalizationLayer& operator=(const BatchNormalizationLayer&);

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
    Matrix& _gamma;
    Matrix& _beta;

private:
    std::unique_ptr<MatrixVector> _internal_parameters;

private:
    Matrix& _means;
    Matrix& _variances;

private:
    size_t _samples;

private:
    std::unique_ptr<matrix::Dimension> _inputSize;

};

}

}



