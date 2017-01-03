/*  \file   MaxPoolingLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the MaxPoolingLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/MaxPoolingLayer.h>

#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
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
: MaxPoolingLayer({1, 1, 1, 1}, {1, 1})
{

}

MaxPoolingLayer::MaxPoolingLayer(const Dimension& inputSize, const Dimension& filterSize)
: MaxPoolingLayer(inputSize, filterSize, matrix::Precision::getDefaultPrecision())
{

}

MaxPoolingLayer::MaxPoolingLayer(const Dimension& inputSize, const Dimension& filterSize,
    const matrix::Precision& precision)
: _inputSize(std::make_unique<matrix::Dimension>(inputSize)),
  _filterSize(std::make_unique<matrix::Dimension>(filterSize)),
  _precision(std::make_unique<matrix::Precision>(precision))
{

}

MaxPoolingLayer::~MaxPoolingLayer()
{
    // intentionally blank
}

MaxPoolingLayer::MaxPoolingLayer(const MaxPoolingLayer& l)
: Layer(l), _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _filterSize(std::make_unique<matrix::Dimension>(*l._filterSize)),
  _precision(std::make_unique<matrix::Precision>(*l._precision))
{

}

MaxPoolingLayer& MaxPoolingLayer::operator=(const MaxPoolingLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _inputSize  = std::make_unique<matrix::Dimension>(*l._inputSize);
    _filterSize = std::make_unique<matrix::Dimension>(*l._filterSize);
    _precision  = std::make_unique<matrix::Precision>(*l._precision);

    return *this;
}

void MaxPoolingLayer::initialize()
{

}

static Matrix setupShape(const Matrix& output, const Dimension& outputSize)
{
    Dimension size = output.size();

    while(size.size() < outputSize.size())
    {
        size.push_back(1);
    }

    return reshape(output, size);
}

void MaxPoolingLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();
    assert(inputActivations.size()[4] == 1);

    util::log("MaxPoolingLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << " through max pooling: "
        << _filterSize->toString() << "\n";

    if(util::isLogEnabled("MaxPoolingLayer::Detail"))
    {
        util::log("MaxPoolingLayer::Detail") << " input: "
            << inputActivations.debugString();
    }

    auto outputActivations = setupShape(getActivationFunction()->apply(
        forwardMaxPooling(inputActivations, *_filterSize)), getOutputSize());

    if(util::isLogEnabled("MaxPoolingLayer::Detail"))
    {
        util::log("MaxPoolingLayer::Detail") << " outputs: "
            << outputActivations.debugString();
    }

    saveMatrix("inputActivations",  inputActivations);
    saveMatrix("outputActivations", outputActivations);

    outputActivationsVector.push_back(std::move(outputActivations));
}

void MaxPoolingLayer::runReverseImplementation(Bundle& bundle)
{
    auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

    auto& inputDeltasVector = bundle["inputDeltas"].get<MatrixVector>();

    // Get the output activations
    auto outputActivations = loadMatrix("outputActivations");

    // Get the input activations and deltas
    auto inputActivations = loadMatrix("inputActivations");

    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations),
        outputDeltas.front(), matrix::Multiply());

    auto inputDeltas = backwardMaxPooling(inputActivations, outputActivations,
        deltas, *_filterSize);

    inputDeltasVector.push_back(std::move(inputDeltas));
}

MatrixVector& MaxPoolingLayer::weights()
{
    static MatrixVector w;
    return w;
}

const MatrixVector& MaxPoolingLayer::weights() const
{
    static MatrixVector w;
    return w;
}

const matrix::Precision& MaxPoolingLayer::precision() const
{
    return *_precision;
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
    auto size = getInputSize();

    size[0] /= (*_filterSize)[0];
    size[1] /= (*_filterSize)[1];

    return size;
}

size_t MaxPoolingLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t MaxPoolingLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t MaxPoolingLayer::totalNeurons() const
{
    return 0;
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
    properties["input-size"]  = _inputSize->toString();
    properties["precision"]   = _precision->toString();

    saveLayer(archive, properties);
}

void MaxPoolingLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    *_filterSize = Dimension::fromString(properties["filter-size"]);
    *_inputSize  = Dimension::fromString(properties["input-size"]);
    _precision   = matrix::Precision::fromString(properties["precision"]);

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> MaxPoolingLayer::clone() const
{
    return std::make_unique<MaxPoolingLayer>(*this);
}

std::string MaxPoolingLayer::getTypeName() const
{
    return "MaxPoolingLayer";
}

}

}





