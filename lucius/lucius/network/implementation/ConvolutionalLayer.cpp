/*  \file   ConvolutionalLayer.cpp
    \author Gregory Diamos
    \date   Dec 24, 2014
    \brief  The source for the ConvolutionalLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/ConvolutionalLayer.h>

#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/ActivationCostFunction.h>
#include <lucius/network/interface/WeightCostFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/ConvolutionalOperations.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/FileOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/TarArchive.h>
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <memory>

namespace lucius
{
namespace network
{

typedef matrix::Matrix       Matrix;
typedef matrix::Dimension    Dimension;
typedef matrix::MatrixVector MatrixVector;

ConvolutionalLayer::ConvolutionalLayer()
: ConvolutionalLayer({}, {}, {}, {}, matrix::SinglePrecision())
{

}

ConvolutionalLayer::~ConvolutionalLayer()
{

}

ConvolutionalLayer::ConvolutionalLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& size, const matrix::Dimension& stride,
    const matrix::Dimension& padding)
: ConvolutionalLayer(inputSize, size, stride, padding, matrix::Precision::getDefaultPrecision())
{

}

static Dimension computeBiasSize(const matrix::Dimension& inputSize,
    const matrix::Dimension& size, const matrix::Dimension& stride, const matrix::Dimension& padding)
{
    auto outputSize = forwardConvolutionOutputSize(inputSize, size, stride, padding);

    return Dimension({outputSize[2]});
}

ConvolutionalLayer::ConvolutionalLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& size, const matrix::Dimension& stride,
    const matrix::Dimension& padding, const matrix::Precision& precision)
: _parameters(std::make_unique<MatrixVector>(
    MatrixVector({Matrix(size, precision),
    Matrix(computeBiasSize(inputSize, size, stride, padding), precision)}))),
  _weights((*_parameters)[0]),
  _bias((*_parameters)[1]),
  _inputSize(std::make_unique<matrix::Dimension>(inputSize)),
  _filterStride(std::make_unique<matrix::Dimension>(stride)),
  _inputPadding(std::make_unique<matrix::Dimension>(padding))
{

}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayer& l)
: Layer(l), _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _weights((*_parameters)[0]),
  _bias((*_parameters)[1]),
  _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _filterStride(std::make_unique<matrix::Dimension>(*l._filterStride)),
  _inputPadding(std::make_unique<matrix::Dimension>(*l._inputPadding))
{

}

ConvolutionalLayer& ConvolutionalLayer::operator=(const ConvolutionalLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _parameters   = std::make_unique<MatrixVector>(*l._parameters);
    _inputSize    = std::make_unique<matrix::Dimension>(*l._inputSize);
    _filterStride = std::make_unique<matrix::Dimension>(*l._filterStride);
    _inputPadding = std::make_unique<matrix::Dimension>(*l._inputPadding);

    return *this;
}

void ConvolutionalLayer::initialize()
{
    auto initializationType = util::KnobDatabase::getKnobValue(
        "ConvolutionalLayer::InitializationType", "glorot");

    if(initializationType == "glorot")
    {
        // Glorot
        double e = util::KnobDatabase::getKnobValue(
            "ConvolutionalLayer::RandomInitializationEpsilon", 6);

        double epsilon = std::sqrt(e) /
            std::sqrt((_weights.size()[0])*(_weights.size()[1])*(_weights.size()[2]) +
            (_weights.size()[3]));

        // generate uniform random values between [0, 1]
        matrix::rand(_weights);

        // shift to center on 0, the range is now [-0.5, 0.5]
        apply(_weights, _weights, matrix::Add(-0.5));

        // scale, the range is now [-epsilon, epsilon]
        apply(_weights, _weights, matrix::Multiply(2.0 * epsilon));
    }
    else if(initializationType == "he")
    {
        // He
        double e = util::KnobDatabase::getKnobValue(
            "ConvolutionalLayer::RandomInitializationEpsilon", 1);

        double epsilon = std::sqrt((2.*e) /
            ((_weights.size()[0])*(_weights.size()[1])*(_weights.size()[2])));

        // generate uniform random values between [0, 1]
        matrix::randn(_weights);

        // scale, the range is now [-epsilon, epsilon]
        apply(_weights, _weights, matrix::Multiply(epsilon));
    }

    // assign bias to 0.0
    apply(_bias, _bias, matrix::Fill(0.0));
}

void ConvolutionalLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivations  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivations = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivations.size() == 1);

    auto m = inputActivations.back();

    saveMatrix("inputActivation", m);

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << " Running forward propagation of matrix "
            << m.shapeString() << " through layer: (size " << _weights.shapeString()
            << ") (stride " << _filterStride->toString() << ") (padding "
            << _inputPadding->toString() << ")\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  input: " << m.debugString();
        util::log("ConvolutionalLayer::Detail") << "  layer: " << _weights.debugString();
        util::log("ConvolutionalLayer::Detail") << "  bias:  " << _bias.debugString();
    }

    auto unbiasedOutput = matrix::forwardConvolution(m, _weights, *_filterStride, *_inputPadding);

    auto output = broadcast(unbiasedOutput, _bias, {0, 1, 3, 4}, matrix::Add());

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  output: " << output.debugString();
    }
    else
    {
        util::log("ConvolutionalLayer") << "  output: " << output.shapeString() << "\n";
    }

    auto activation = getActivationFunction()->apply(output);

    saveMatrix("outputActivation", copy(activation));

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  activation: " << activation.debugString();
    }
    else
    {
        util::log("ConvolutionalLayer") << "  activation: " << activation.shapeString() << "\n";
    }

    outputActivations.push_back(std::move(activation));
}

void ConvolutionalLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients    = bundle[   "gradients"].get<MatrixVector>();
    auto& inputDeltas  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

    auto outputActivation = loadMatrix("outputActivation");
    auto inputActivation  = loadMatrix("inputActivation");

    assert(outputDeltas.size() == 1);

    auto difference = outputDeltas.front();

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << " Running reverse propagation on matrix ("
            << difference.shapeString() << " columns) through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("ConvolutionalLayer") << "  layer: " << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  difference size: "
            << difference.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  difference: " << difference.debugString();
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  output activations size: "
            << outputActivation.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  output activations: "
            << outputActivation.debugString();
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  input activations size: "
            << inputActivation.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  input activations: "
            << inputActivation.debugString();
    }

    // finish computing the deltas
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivation),
        difference, matrix::Multiply());

    size_t samples = deltas.size()[3];

    assert(samples != 0);

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // compute gradient for the weights
    Matrix weightGradient(_weights.size(), _weights.precision());

    reverseConvolutionGradients(weightGradient, inputActivation, deltas,
        *_filterStride, *_inputPadding, 1.0 / samples);

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradient, weightGradient,
            getWeightCostFunction()->getGradient(_weights), matrix::Add());
    }

    gradients.push_back(std::move(weightGradient));

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  weight grad shape: "
            << weightGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  weight grad: "
            << weightGradient.debugString();
    }

    // compute gradient for the bias
    auto biasGradient = reduce(apply(deltas, matrix::Divide(samples)),
        {0, 1, 3, 4}, matrix::Add());

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  bias grad shape: "
            << biasGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  bias grad: "
            << biasGradient.debugString();
    }

    assert(biasGradient.size() == _bias.size());

    gradients.push_back(std::move(biasGradient));

    // compute deltas for previous layer
    Matrix deltasPropagatedReverse(inputActivation.size(), inputActivation.precision());

    if(getShouldComputeDeltas())
    {
        reverseConvolutionDeltas(deltasPropagatedReverse, _weights,
            *_filterStride, deltas, *_inputPadding);
    }

    Matrix previousLayerDeltas;

    if(getActivationCostFunction() != nullptr)
    {
        auto activationCostFunctionGradient =
            getActivationCostFunction()->getGradient(outputActivation);

        apply(previousLayerDeltas, deltasPropagatedReverse,
            activationCostFunctionGradient, matrix::Multiply());
    }
    else
    {
        previousLayerDeltas = std::move(deltasPropagatedReverse);
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  previous layer deltas shape: "
            << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  previous layer deltas: "
            << previousLayerDeltas.debugString();
    }

    inputDeltas.push_back(std::move(previousLayerDeltas));
}

MatrixVector& ConvolutionalLayer::weights()
{
    return *_parameters;
}

const MatrixVector& ConvolutionalLayer::weights() const
{
    return *_parameters;
}

const matrix::Precision& ConvolutionalLayer::precision() const
{
    return _weights.precision();
}

double ConvolutionalLayer::computeWeightCost() const
{
    return getWeightCostFunction()->getCost(_weights);
}

Dimension ConvolutionalLayer::getInputSize() const
{
    return *_inputSize;
}

Dimension ConvolutionalLayer::getOutputSize() const
{
    return forwardConvolutionOutputSize(*_inputSize, _weights.size(),
        *_filterStride, *_inputPadding);
}

size_t ConvolutionalLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t ConvolutionalLayer::getOutputCount() const
{
    auto size = getOutputSize();

    size.pop_back();

    return size.product();
}

size_t ConvolutionalLayer::totalNeurons() const
{
    return getOutputCount();
}

size_t ConvolutionalLayer::totalConnections() const
{
    return _weights.elements();
}

size_t ConvolutionalLayer::getFloatingPointOperationCount() const
{
    size_t filterOps = _weights.size()[0] * _weights.size()[1] * _weights.size()[3];

    return 2 * getInputSize().product() * filterOps / _filterStride->product() +
        _bias.elements();
}

size_t ConvolutionalLayer::getActivationMemory() const
{
    return precision().size() * getOutputSize().product();
}

void ConvolutionalLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["weights"] = properties.path() + "." + properties.key() + ".weights.npy";
    properties["bias"]    = properties.path() + "." + properties.key() + ".bias.npy";

    properties["input-size"]    = _inputSize->toString();
    properties["filter-stride"] = _filterStride->toString();
    properties["input-padding"] = _inputPadding->toString();

    saveToArchive(archive, properties["weights"], _weights);
    saveToArchive(archive, properties["bias"],    _bias);

    saveLayer(archive, properties);
}

void ConvolutionalLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    _weights = matrix::loadFromArchive(archive, properties["weights"]);
    _bias    = matrix::loadFromArchive(archive, properties["bias"]);

    *_inputSize    = Dimension::fromString(properties["input-size"]);
    *_filterStride = Dimension::fromString(properties["filter-stride"]);
    *_inputPadding = Dimension::fromString(properties["input-padding"]);

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> ConvolutionalLayer::clone() const
{
    return std::make_unique<ConvolutionalLayer>(*this);
}

std::string ConvolutionalLayer::getTypeName() const
{
    return "ConvolutionalLayer";
}

matrix::Dimension ConvolutionalLayer::getFilterStride() const
{
    return *_filterStride;
}

matrix::Dimension ConvolutionalLayer::getInputPadding() const
{
    return *_inputPadding;
}

}

}





