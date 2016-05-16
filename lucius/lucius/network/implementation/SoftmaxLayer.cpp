/*  \file   SoftmaxLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the SoftmaxLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/SoftmaxLayer.h>

#include <lucius/network/interface/ActivationFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>

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

SoftmaxLayer::SoftmaxLayer()
: SoftmaxLayer({1, 1, 1})
{

}

SoftmaxLayer::SoftmaxLayer(const Dimension& inputSize)
: SoftmaxLayer(inputSize, matrix::Precision::getDefaultPrecision())
{

}

SoftmaxLayer::SoftmaxLayer(const Dimension& inputSize,
    const matrix::Precision& precision)
: _inputSize(std::make_unique<matrix::Dimension>(inputSize)),
  _precision(std::make_unique<matrix::Precision>(precision))
{

}

SoftmaxLayer::~SoftmaxLayer()
{

}

SoftmaxLayer::SoftmaxLayer(const SoftmaxLayer& l)
: Layer(l), _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _precision(std::make_unique<matrix::Precision>(*l._precision))
{

}

SoftmaxLayer& SoftmaxLayer::operator=(const SoftmaxLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _inputSize  = std::make_unique<matrix::Dimension>(*l._inputSize);
    _precision  = std::make_unique<matrix::Precision>(*l._precision);

    return *this;
}

void SoftmaxLayer::initialize()
{

}

static Matrix foldTime(const Matrix& input)
{
    assert(input.size().size() < 4);

    if(input.size().size() == 3)
    {
        auto size = input.size();
        size_t timesteps = size.back();

        size.pop_back();

        size.back() *= timesteps;

        return reshape(input, size);
    }

    return input;
}

static Matrix unfoldTime(const Matrix& result, const Dimension& inputSize)
{
    if(inputSize.size() <= 2)
    {
        return result;
    }

    assert(inputSize.size() == 3);

    size_t layerSize = result.size()[0];
    size_t miniBatch = inputSize[1];
    size_t timesteps = inputSize[2];

    return reshape(result, {layerSize, miniBatch, timesteps});
}

void SoftmaxLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = foldTime(inputActivationsVector.back());

    util::log("SoftmaxLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << "\n";

    if(util::isLogEnabled("SoftmaxLayer::Detail"))
    {
        util::log("SoftmaxLayer::Detail") << " input: "
            << inputActivations.debugString();
    }

    auto outputActivations = softmax(inputActivations);

    if(util::isLogEnabled("SoftmaxLayer::Detail"))
    {
        util::log("SoftmaxLayer::Detail") << " outputs: "
            << outputActivations.debugString();
    }

    saveMatrix("outputActivations", outputActivations);

    outputActivationsVector.push_back(unfoldTime(outputActivations,
        inputActivationsVector.front().size()));
}

void SoftmaxLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients          = bundle[   "gradients"].get<MatrixVector>();
    auto& inputDeltasVector  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

    // Get the output activations
    auto outputActivations = loadMatrix("outputActivations");

    // Get the output deltas
    auto outputDeltas = foldTime(outputDeltasVector.front());

    auto sum = reduce(apply(matrix::Matrix(outputActivations), outputDeltas, matrix::Multiply()),
        {0}, matrix::Add());

    auto inputDeltas = apply(broadcast(outputDeltas, sum, {0}, matrix::Subtract()),
        outputActivations, matrix::Multiply());

    inputDeltasVector.push_back(unfoldTime(inputDeltas, outputDeltasVector.front().size()));
}

MatrixVector& SoftmaxLayer::weights()
{
    static MatrixVector w;
    return w;
}

const MatrixVector& SoftmaxLayer::weights() const
{
    static MatrixVector w;

    return w;
}

const matrix::Precision& SoftmaxLayer::precision() const
{
    return *_precision;
}

double SoftmaxLayer::computeWeightCost() const
{
    return 0.0;
}

Dimension SoftmaxLayer::getInputSize() const
{
    return *_inputSize;
}

Dimension SoftmaxLayer::getOutputSize() const
{
    return getInputSize();
}

size_t SoftmaxLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t SoftmaxLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t SoftmaxLayer::totalNeurons() const
{
    return 0;
}

size_t SoftmaxLayer::totalConnections() const
{
    return totalNeurons();
}

size_t SoftmaxLayer::getFloatingPointOperationCount() const
{
    return 2 * getInputCount();
}

size_t SoftmaxLayer::getActivationMemory() const
{
    return precision().size() * getOutputCount();
}

void SoftmaxLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["input-size"]  = _inputSize->toString();
    properties["precision"]   = _precision->toString();

    saveLayer(archive, properties);
}

void SoftmaxLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    *_inputSize  = Dimension::fromString(properties["input-size"]);
    _precision   = matrix::Precision::fromString(properties["precision"]);

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> SoftmaxLayer::clone() const
{
    return std::make_unique<SoftmaxLayer>(*this);
}

std::string SoftmaxLayer::getTypeName() const
{
    return "SoftmaxLayer";
}

}

}






