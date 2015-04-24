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

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;

ConvolutionalLayer::ConvolutionalLayer()
: ConvolutionalLayer({}, {}, {}, matrix::SinglePrecision())
{

}

ConvolutionalLayer::~ConvolutionalLayer()
{

}

ConvolutionalLayer::ConvolutionalLayer(const matrix::Dimension& inputSize,
    const matrix::Dimension& size, const matrix::Dimension& stride, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix(size, precision), Matrix({size[0]}, precision)})), //TODO: fix this bias!
  _weights((*_parameters)[0]),
  _bias((*_parameters)[1]),
  _inputSize(std::make_unique<matrix::Dimension>(inputSize)),
  _filterStride(std::make_unique<matrix::Dimension>(stride))
{

}

ConvolutionalLayer::ConvolutionalLayer(const ConvolutionalLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _weights((*_parameters)[0]),
  _bias((*_parameters)[1]),
  _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _filterStride(std::make_unique<matrix::Dimension>(*l._filterStride))
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

Matrix ConvolutionalLayer::runForward(const Matrix& m) const
{
    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << " Running forward propagation through layer: " << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  input: " << m.debugString();
        util::log("ConvolutionalLayer::Detail") << "  layer: " << _weights.debugString();
        util::log("ConvolutionalLayer::Detail") << "  bias:  " << _bias.debugString();
    }

    auto unbiasedOutput = matrix::forwardConvolution(m, _weights, *_filterStride);

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

    return activation;
}

Matrix ConvolutionalLayer::runReverse(MatrixVector& gradients,
    const Matrix& inputActivations,
    const Matrix& outputActivations,
    const Matrix& difference) const
{
    if(util::isLogEnabled("ConvolutionalLayer"))
    {
        util::log("ConvolutionalLayer") << " Running reverse propagation on matrix (" << difference.size()[0]
            << " rows, " << difference.size()[1] << " columns) through layer with dimensions ("
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
        util::log("ConvolutionalLayer") << "  output size: " << outputActivations.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  output: " << outputActivations.debugString();
    }

    // finish computing the deltas
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations), difference, matrix::Multiply());

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // compute gradient for the weights
    auto weightGradient = reverseConvolutionGradients(_weights, inputActivations, deltas);

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
    auto samples      = deltas.size()[1];
    auto biasGradient = reduce(apply(deltas, matrix::Divide(samples)), {1}, matrix::Add());

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
    auto deltasPropagatedReverse = reverseConvolutionDeltas(_weights, deltas);

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
        util::log("ConvolutionalLayer") << "  output shape: " << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("ConvolutionalLayer::Detail"))
    {
        util::log("ConvolutionalLayer::Detail") << "  output: " << previousLayerDeltas.debugString();
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

size_t ConvolutionalLayer::getInputCount() const
{
    return _weights.size()[0] * _weights.size()[1] * _weights.size()[2];
}

size_t ConvolutionalLayer::getOutputCount() const
{
    return _weights.size()[3];
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
        *_filterStride, precision());
}

std::string ConvolutionalLayer::getTypeName() const
{
    return "ConvolutionalLayer";
}

}

}





