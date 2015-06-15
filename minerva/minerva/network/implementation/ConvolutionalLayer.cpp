/*  \file   ConvolutionalLayer.cpp
    \author Gregory Diamos
     \date   Dec 24, 2014
     \brief  The source for the ConvolutionalLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/ConvolutionalLayer.h>

#include <minerva/network/interface/ActivationFunction.h>
#include <minerva/network/interface/ActivationCostFunction.h>
#include <minerva/network/interface/WeightCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/BlasOperations.h>
#include <minerva/matrix/interface/ConvolutionalOperations.h>
#include <minerva/matrix/interface/RandomOperations.h>
#include <minerva/matrix/interface/Operation.h>
#include <minerva/matrix/interface/MatrixVector.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/knobs.h>
#include <minerva/util/interface/memory.h>

// Standard Library Includes
#include <memory>

namespace minerva
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
    const matrix::Dimension& size, const matrix::Dimension& stride, const matrix::Dimension& padding)
: ConvolutionalLayer(inputSize, size, stride, padding, matrix::Precision::getDefaultPrecision())
{

}

static Dimension computeBiasSize(const matrix::Dimension& inputSize,
    const matrix::Dimension& size, const matrix::Dimension& stride, const matrix::Dimension& padding)
{
    auto outputSize = forwardConvolutionOutputSize(inputSize, size, stride, padding);

    while(outputSize.size() > 3)
    {
        outputSize.pop_back();
    }

    return outputSize;
}

ConvolutionalLayer::ConvolutionalLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& size, const matrix::Dimension& stride, const matrix::Dimension& padding, const matrix::Precision& precision)
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
: _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _weights((*_parameters)[0]),
  _bias((*_parameters)[1]),
  _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _filterStride(std::make_unique<matrix::Dimension>(*l._filterStride)),
  _inputPadding(std::make_unique<matrix::Dimension>(*l._inputPadding))
{

}

ConvolutionalLayer& ConvolutionalLayer::operator=(const ConvolutionalLayer& l)
{
    if(&l == this)
    {
        return *this;
    }

    _parameters   = std::move(std::make_unique<MatrixVector>(*l._parameters));
    _inputSize    = std::move(std::make_unique<matrix::Dimension>(*l._inputSize));
    _filterStride = std::move(std::make_unique<matrix::Dimension>(*l._filterStride));
    _inputPadding = std::move(std::make_unique<matrix::Dimension>(*l._inputPadding));

    return *this;
}

void ConvolutionalLayer::initialize()
{
    double e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", 6);

    double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));

    // generate uniform random values between [0, 1]
    matrix::rand(_weights);

    // shift to center on 0, the range is now [-0.5, 0.5]
    apply(_weights, _weights, matrix::Add(-0.5));

    // scale, the range is now [-epsilon, epsilon]
    apply(_weights, _weights, matrix::Multiply(2.0 * epsilon));

    // assign bias to 0.0
    apply(_bias, _bias, matrix::Fill(0.0));
}

void ConvolutionalLayer::runForwardImplementation(MatrixVector& activations) const
{
    auto m = activations.back();

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << " Running forward propagation of matrix " << m.shapeString()
            << " through layer: (size " << _weights.shapeString()
            << ") (stride " << _filterStride->toString() << ") (padding " << _inputPadding->toString() << ")\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  input: " << m.debugString();
        util::log("ConvolutionalLayer::Detail") << "  layer: " << _weights.debugString();
        util::log("ConvolutionalLayer::Detail") << "  bias:  " << _bias.debugString();
    }

    auto unbiasedOutput = matrix::forwardConvolution(m, _weights, *_filterStride, *_inputPadding);

    auto output = broadcast(unbiasedOutput, _bias, {}, matrix::Add());

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  output: " << output.debugString();
    }
    else
    {
        util::log("ConvolutionalLayer") << "  output: " << output.shapeString() << "\n";
    }

    auto activation = getActivationFunction()->apply(output);

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  activation: " << activation.debugString();
    }
    else
    {
        util::log("ConvolutionalLayer") << "  activation: " << activation.shapeString() << "\n";
    }

    activations.push_back(std::move(activation));
}

Matrix ConvolutionalLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& difference) const
{
    auto outputActivations = activations.back();
    activations.pop_back();

    auto inputActivations = activations.back();

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << " Running reverse propagation on matrix ("
            << difference.shapeString() << " columns) through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("ConvolutionalLayer") << "  layer: " << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  difference size: " << difference.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  difference: " << difference.debugString();
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  output activations size: " << outputActivations.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  output activations: " << outputActivations.debugString();
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  input activations size: " << inputActivations.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  input activations: " << inputActivations.debugString();
    }

    // finish computing the deltas
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations), difference, matrix::Multiply());

    size_t samples = deltas.size()[3];

    assert(samples != 0);

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // compute gradient for the weights
    Matrix weightGradient(_weights.size(), _weights.precision());
   
    reverseConvolutionGradients(weightGradient, inputActivations, deltas, *_filterStride, *_inputPadding, 1.0 / samples);

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradient, weightGradient, getWeightCostFunction()->getGradient(_weights), matrix::Add());
    }

    gradients.push_back(std::move(weightGradient));

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  weight grad shape: " << weightGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  weight grad: " << weightGradient.debugString();
    }

    // compute gradient for the bias
    auto biasGradient = reduce(apply(deltas, matrix::Divide(samples)), {3, 4}, matrix::Add());

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  bias grad shape: " << biasGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  bias grad: " << biasGradient.debugString();
    }

    assert(biasGradient.size() == _bias.size());

    gradients.push_back(std::move(biasGradient));

    // compute deltas for previous layer
    Matrix deltasPropagatedReverse(inputActivations.size(), inputActivations.precision());
    
    reverseConvolutionDeltas(deltasPropagatedReverse, _weights, *_filterStride, deltas, *_inputPadding);

    Matrix previousLayerDeltas;

    if(getActivationCostFunction() != nullptr)
    {
        auto activationCostFunctionGradient = getActivationCostFunction()->getGradient(outputActivations);

        apply(previousLayerDeltas, deltasPropagatedReverse, activationCostFunctionGradient, matrix::Multiply());
    }
    else
    {
        previousLayerDeltas = std::move(deltasPropagatedReverse);
    }

    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << "  previous layer deltas shape: " << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  previous layer deltas: " << previousLayerDeltas.debugString();
    }

    return previousLayerDeltas;
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
    return forwardConvolutionOutputSize(*_inputSize, _weights.size(), *_filterStride, *_inputPadding);
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
    return 2 * totalConnections();
}

void ConvolutionalLayer::save(util::TarArchive& archive) const
{
    assertM(false, "Not implemented.");
}

void ConvolutionalLayer::load(const util::TarArchive& archive, const std::string& name)
{
    assertM(false, "Not implemented.");
}

std::unique_ptr<Layer> ConvolutionalLayer::clone() const
{
    return std::make_unique<ConvolutionalLayer>(*this);
}

std::unique_ptr<Layer> ConvolutionalLayer::mirror() const
{
    return std::make_unique<ConvolutionalLayer>(*_inputSize,
        matrix::Dimension(_weights.size()[0], _weights.size()[1], _weights.size()[3], _weights.size()[2]),
        *_filterStride, *_inputPadding, precision());
}

std::string ConvolutionalLayer::getTypeName() const
{
    return "ConvolutionalLayer";
}

}

}





