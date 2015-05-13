/*  \file   ConvolutionalLayer.h
    \author Gregory Diamos
     \date   Dec 24, 2014
     \brief  The interface for the ConvolutionalLayer class.
*/

#pragma once

// Minerva Includes
#include <minerva/network/interface/Layer.h>

namespace minerva
{
namespace network
{

/* \brief An implementation of a generic recurrent layer. */
class ConvolutionalLayer : public Layer
{
public:
    ConvolutionalLayer();
    virtual ~ConvolutionalLayer();

public:
    ConvolutionalLayer(const matrix::Dimension& inputSize, const matrix::Dimension& filterSize,
        const matrix::Dimension& filterStride, const matrix::Dimension& inputPadding);
    ConvolutionalLayer(const matrix::Dimension& inputSize, const matrix::Dimension& filterSize,
        const matrix::Dimension& filterStride, const matrix::Dimension& inputPadding, const matrix::Precision&);

public:
    ConvolutionalLayer(const ConvolutionalLayer& );
    ConvolutionalLayer& operator=(const ConvolutionalLayer&);

public:
    virtual void initialize();

public:
    virtual Matrix runForward(const Matrix& m) const;
    virtual Matrix runReverse(MatrixVector& gradients,
        const Matrix& inputActivations,
        const Matrix& outputActivations,
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

private:
    std::unique_ptr<matrix::Dimension> _inputSize;
    std::unique_ptr<matrix::Dimension> _filterStride;
    std::unique_ptr<matrix::Dimension> _inputPadding;

};
}

}




