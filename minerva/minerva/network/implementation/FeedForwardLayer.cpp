/*  \file   FeedForwardLayer.h
    \author Gregory Diamos
     \date   Dec 24, 2014
     \brief  The implementation of the FeedForwardLayer class.
*/

// Minerva Includes
#include <minerva/network/interface/FeedForwardLayer.h>

#include <minerva/network/interface/ActivationFunction.h>
#include <minerva/network/interface/ActivationCostFunction.h>
#include <minerva/network/interface/WeightCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/BlasOperations.h>
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
typedef matrix::MatrixVector MatrixVector;
typedef matrix::Dimension    Dimension;

FeedForwardLayer::FeedForwardLayer()
: FeedForwardLayer(0, 0, matrix::SinglePrecision())
{

}

FeedForwardLayer::FeedForwardLayer(size_t inputs, size_t outputs)
: FeedForwardLayer(inputs, outputs, matrix::Precision::getDefaultPrecision())
{

}

FeedForwardLayer::FeedForwardLayer(size_t inputs, size_t outputs, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({outputs, inputs}, precision), Matrix({outputs}, precision)})),
 _weights((*_parameters)[0]), _bias((*_parameters)[1])
{

}

FeedForwardLayer::~FeedForwardLayer()
{

}

FeedForwardLayer::FeedForwardLayer(const FeedForwardLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)), _weights((*_parameters)[0]), _bias((*_parameters)[1])
{

}

FeedForwardLayer& FeedForwardLayer::operator=(const FeedForwardLayer& l)
{
    if(&l == this)
    {
        return *this;
    }

    _parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));

    return *this;
}

void FeedForwardLayer::initialize()
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

void FeedForwardLayer::runForwardImplementation(MatrixVector& activations) const
{
    auto m = foldTime(activations.back());

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << " Running forward propagation of matrix " << m.shapeString()
            << " through layer: " << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  input: " << m.debugString();
        util::log("FeedForwardLayer::Detail") << "  layer: " << _weights.debugString();
        util::log("FeedForwardLayer::Detail") << "  bias:  " << _bias.debugString();
    }
    else
    {
        util::log("FeedForwardLayer") << "  input shape: " << activations.back().shapeString() << "\n";
    }

    auto unbiasedOutput = gemm(Matrix(_weights), false, 1.0, m, false);

    auto output = broadcast(unbiasedOutput, _bias, {}, matrix::Add());

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  output: " << output.debugString();
    }
    else
    {
        util::log("FeedForwardLayer") << "  output shape: " << output.shapeString() << "\n";
    }

    auto activation = unfoldTime(getActivationFunction()->apply(output), activations.back().size());

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  activation: " << activation.debugString();
    }
    else
    {
        util::log("FeedForwardLayer") << "  activation: " << activation.shapeString() << "\n";
    }

    activations.push_back(std::move(activation));
}

matrix::Matrix FeedForwardLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& differenceWithTime) const
{
    auto outputActivations = foldTime(activations.back());
    activations.pop_back();

    auto inputActivations = foldTime(activations.back());
    auto difference = foldTime(differenceWithTime);

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << " Running reverse propagation on matrix (" << difference.size()[0]
            << " rows, " << difference.size()[1] << " columns) through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("FeedForwardLayer") << "  layer: " << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  difference size: " << difference.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  difference: " << difference.debugString();
    }

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  output size: " << outputActivations.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  output: " << outputActivations.debugString();
    }

    // finish computing the deltas
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivations), difference, matrix::Multiply());

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // compute gradient for the weights
    auto samples = outputActivations.size()[1];

    auto weightGradient = gemm(Matrix(deltas), false, 1.0 / samples, inputActivations, true);

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradient, weightGradient, getWeightCostFunction()->getGradient(_weights), matrix::Add());
    }

    gradients.push_back(std::move(weightGradient));

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  weight grad shape: " << weightGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  weight grad: " << weightGradient.debugString();
    }

    // compute gradient for the bias
    auto biasGradient = reduce(apply(deltas, matrix::Divide(samples)), {1}, matrix::Add());

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  bias grad shape: " << biasGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  bias grad: " << biasGradient.debugString();
    }

    assert(biasGradient.size() == _bias.size());

    gradients.push_back(std::move(biasGradient));

    // compute deltas for previous layer
    auto deltasPropagatedReverse = gemm(_weights, true, deltas, false);

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

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  output shape: " << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  output: " << previousLayerDeltas.debugString();
    }

    return unfoldTime(previousLayerDeltas, differenceWithTime.size());
}

MatrixVector& FeedForwardLayer::weights()
{
    return *_parameters;
}

const MatrixVector& FeedForwardLayer::weights() const
{
    return *_parameters;
}

const matrix::Precision& FeedForwardLayer::precision() const
{
    return _weights.precision();
}

double FeedForwardLayer::computeWeightCost() const
{
    return getWeightCostFunction()->getCost(_weights);
}

Dimension FeedForwardLayer::getInputSize() const
{
    return {getInputCount(), 1, 1};
}

Dimension FeedForwardLayer::getOutputSize() const
{
    return {getOutputCount(), 1, 1};
}

size_t FeedForwardLayer::getInputCount() const
{
    return _weights.size()[1];
}

size_t FeedForwardLayer::getOutputCount() const
{
    return _weights.size()[0];
}

size_t FeedForwardLayer::totalNeurons()    const
{
    return getOutputCount();
}

size_t FeedForwardLayer::totalConnections() const
{
    return _weights.elements() + _bias.elements();
}

size_t FeedForwardLayer::getFloatingPointOperationCount() const
{
    return 2 * totalConnections();
}

void FeedForwardLayer::save(util::TarArchive& archive) const
{
    assertM(false, "Not implemented");
}

void FeedForwardLayer::load(const util::TarArchive& archive, const std::string& name)
{
    assertM(false, "Not implemented");
}

std::unique_ptr<Layer> FeedForwardLayer::clone() const
{
    return std::make_unique<FeedForwardLayer>(*this);
}

std::unique_ptr<Layer> FeedForwardLayer::mirror() const
{
    return std::make_unique<FeedForwardLayer>(getInputCount(), getOutputCount(), precision());
}

std::string FeedForwardLayer::getTypeName() const
{
    return "FeedForwardLayer";
}

}

}


