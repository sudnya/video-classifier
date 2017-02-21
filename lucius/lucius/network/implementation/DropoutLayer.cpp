/*  \file   DropoutLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the DropoutLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/DropoutLayer.h>

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

DropoutLayer::DropoutLayer()
: DropoutLayer({1, 1, 1, 1})
{

}

DropoutLayer::DropoutLayer(const Dimension& inputSize)
: DropoutLayer(inputSize, matrix::Precision::getDefaultPrecision())
{

}

DropoutLayer::DropoutLayer(const Dimension& inputSize, const matrix::Precision& precision)
: _inputSize(std::make_unique<matrix::Dimension>(inputSize)),
  _precision(std::make_unique<matrix::Precision>(precision)),
  _trainingIteration(0)
{

}

DropoutLayer::~DropoutLayer()
{
    // intentionally blank
}

DropoutLayer::DropoutLayer(const DropoutLayer& l)
: Layer(l),
  _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _precision(std::make_unique<matrix::Precision>(*l._precision)),
  _trainingIteration(l._trainingIteration)
{

}

DropoutLayer& DropoutLayer::operator=(const DropoutLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _inputSize  = std::make_unique<matrix::Dimension>(*l._inputSize);
    _precision  = std::make_unique<matrix::Precision>(*l._precision);
    _trainingIteration = l._trainingIteration;

    return *this;
}

void DropoutLayer::initialize()
{

}

void DropoutLayer::runForwardImplementation(Bundle& bundle)
{
    // TODO
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    util::log("DropoutLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << "\n";

    if(util::isLogEnabled("DropoutLayer::Detail"))
    {
        util::log("DropoutLayer::Detail") << " input: "
            << inputActivations.debugString();
    }

    auto outputActivations = getActivationFunction()->apply(
        forwardDropout(inputActivations, *_filterSize));

    if(util::isLogEnabled("DropoutLayer::Detail"))
    {
        util::log("DropoutLayer::Detail") << " outputs: "
            << outputActivations.debugString();
    }

    saveMatrix("inputActivations",  inputActivations);
    saveMatrix("outputActivations", outputActivations);

    outputActivationsVector.push_back(std::move(outputActivations));
}

void DropoutLayer::runReverseImplementation(Bundle& bundle)
{
    auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

    auto& inputDeltasVector = bundle["inputDeltas"].get<MatrixVector>();

    // Get the output activations
    auto outputActivations = loadMatrix("outputActivations");

    // Get the input activations and deltas
    auto inputActivations = loadMatrix("inputActivations");

    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations),
        outputDeltas.front(), matrix::Multiply());

    auto inputDeltas = backwardDropout(inputActivations, outputActivations,
        deltas, *_filterSize);

    inputDeltasVector.push_back(std::move(inputDeltas));
}

MatrixVector& DropoutLayer::weights()
{
    static MatrixVector w;
    return w;
}

const MatrixVector& DropoutLayer::weights() const
{
    static MatrixVector w;
    return w;
}

const matrix::Precision& DropoutLayer::precision() const
{
    return *_precision;
}

double DropoutLayer::computeWeightCost() const
{
    return 0.0;
}

Dimension DropoutLayer::getInputSize() const
{
    return *_inputSize;
}

Dimension DropoutLayer::getOutputSize() const
{
    return getInputSize();
}

size_t DropoutLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t DropoutLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t DropoutLayer::totalNeurons() const
{
    return 0;
}

size_t DropoutLayer::totalConnections() const
{
    return totalNeurons();
}

size_t DropoutLayer::getFloatingPointOperationCount() const
{
    return getInputCount();
}

size_t DropoutLayer::getActivationMemory() const
{
    return precision().size() * getOutputCount();
}

void DropoutLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["input-size"] = _inputSize->toString();
    properties["precision"]  = _precision->toString();

    properties["training-iteration"] = _trainingIteration;

    saveLayer(archive, properties);
}

void DropoutLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    *_inputSize = Dimension::fromString(properties["input-size"]);
    _precision  = matrix::Precision::fromString(properties["precision"]);

    _iteratingIteration = properties.get<size_t>("training-iteration");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> DropoutLayer::clone() const
{
    return std::make_unique<DropoutLayer>(*this);
}

std::string DropoutLayer::getTypeName() const
{
    return "DropoutLayer";
}

}

}






