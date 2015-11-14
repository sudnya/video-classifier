/*  \file   MaxPoolingLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the MaxPoolingLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/MaxPoolingLayer.h>

#include <lucius/network/interface/ActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/FileOperations.h>
#include <lucius/matrix/interface/PoolingOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/PropertyTree.h>

namespace lucius
{
namespace network
{

typedef matrix::Matrix       Matrix;
typedef matrix::Dimension    Dimension;
typedef matrix::MatrixVector MatrixVector;

MaxPoolingLayer::MaxPoolingLayer()
: MaxPoolingLayer({1, 1})
{

}

MaxPoolingLayer::MaxPoolingLayer(const Dimension& size)
: MaxPoolingLayer(size, matrix::Precision::getDefaultPrecision())
{

}

MaxPoolingLayer::MaxPoolingLayer(const Dimension& size,
    const matrix::Precision& precision)
: _filterSize(std::make_unique<matrix::Dimension>(size))
{

}

MaxPoolingLayer::~MaxPoolingLayer()
{

}

MaxPoolingLayer::MaxPoolingLayer(const MaxPoolingLayer& l)
: _filterSize(std::make_unique<matrix::Dimension>(*l._filterSize))
{

}

MaxPoolingLayer& MaxPoolingLayer::operator=(const MaxPoolingLayer& l)
{
    if(&l == this)
    {
        return *this;
    }

    _filterSize  = std::move(std::make_unique<matrix::Dimension>(*l._inputSize));

    return *this;
}

void MaxPoolingLayer::initialize()
{

}

static Matrix foldTime(const Matrix& input)
{
    auto size = input.size();

    size_t batch = 1;

    for(size_t i = 1; i < size.size(); ++i)
    {
        batch *= size[i];
    }

    return reshape(input, {size[0], batch});
}

static Matrix unfoldTime(const Matrix& result, const Dimension& inputSize)
{
    return reshape(result, inputSize);
}

void MaxPoolingLayer::runForwardImplementation(MatrixVector& activations)
{
    auto inputActivations = foldTime(activations.back());

    util::log("MaxPoolingLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << " through max pooling: "
        << _filterSize->toString() << "\n";

    if(util::isLogEnabled("MaxPoolingLayer::Detail"))
    {
        util::log("MaxPoolingLayer::Detail") << " input: "
            << inputActivations.debugString();
    }

    auto outputActivations = unfoldTime(getActivationFunction()->apply(
        forwardMaxPooling(activations, *_filterSize)), activations.back().size());

    if(util::isLogEnabled("MaxPoolingLayer::Detail"))
    {
        util::log("MaxPoolingLayer::Detail") << " outputs: "
            << outputActivations.debugString();
    }

    activations.push_back(std::move(outputActivations));
}

Matrix MaxPoolingLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& deltasWithTime)
{
    // Get the output activations
    auto outputActivations = foldTime(activations.back());

    // deallocate memory for the output activations
    activations.pop_back();

    // Get the input activations and deltas
    auto inputActivations = foldTime(activations.back());
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations),
        foldTime(deltasWithTime), matrix::Multiply());

    auto inputDeltas = backwardMaxPooling(inputActivations, outputActivations,
        deltas, *_filterShape);

    return unfoldTime(inputDeltas, deltasWithTime.size());
}

MatrixVector& MaxPoolingLayer::weights()
{
    return *_parameters;
}

const MatrixVector& MaxPoolingLayer::weights() const
{
    return *_parameters;
}

const matrix::Precision& MaxPoolingLayer::precision() const
{
    return _gamma.precision();
}

double MaxPoolingLayer::computeWeightCost() const
{
    return 0.0;
}

Dimension MaxPoolingLayer::getInputSize() const
{
    return *_inputSize;
}

Dimension MaxPoolingLayer::getOutputSize() const
{
    return getInputSize();
}

size_t MaxPoolingLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t MaxPoolingLayer::getOutputCount() const
{
    return getInputCount();
}

size_t MaxPoolingLayer::totalNeurons() const
{
    return _gamma.elements();
}

size_t MaxPoolingLayer::totalConnections() const
{
    return totalNeurons();
}

size_t MaxPoolingLayer::getFloatingPointOperationCount() const
{
    return 2 * getInputCount();
}

size_t MaxPoolingLayer::getActivationMemory() const
{
    return precision().size() * getOutputCount();
}

void MaxPoolingLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["filter-size"] = _filterSize->toString();

    saveLayer(archive, properties);
}

void MaxPoolingLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    *_filterSize = Dimension::fromString(properties["filter-size"]);

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> MaxPoolingLayer::clone() const
{
    return std::make_unique<MaxPoolingLayer>(*this);
}

std::unique_ptr<Layer> MaxPoolingLayer::mirror() const
{
    return clone();
}

std::string MaxPoolingLayer::getTypeName() const
{
    return "MaxPoolingLayer";
}

}

}





