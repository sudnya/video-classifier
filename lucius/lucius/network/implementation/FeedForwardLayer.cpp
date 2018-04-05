/*  \file   FeedForwardLayer.h
    \author Gregory Diamos
    \date   Dec 24, 2014
    \brief  The implementation of the FeedForwardLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/FeedForwardLayer.h>

#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/ActivationCostFunction.h>
#include <lucius/network/interface/WeightCostFunction.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/FileOperations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <memory>

namespace lucius
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

FeedForwardLayer::FeedForwardLayer(size_t inputs, size_t outputs,
    const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({outputs, inputs}, precision),
                                Matrix({outputs},         precision)})),
  _weights((*_parameters)[0]),
  _bias((*_parameters)[1])
{

}

FeedForwardLayer::~FeedForwardLayer()
{

}

FeedForwardLayer::FeedForwardLayer(const FeedForwardLayer& l)
: Layer(l), _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _weights((*_parameters)[0]), _bias((*_parameters)[1])
{

}

FeedForwardLayer& FeedForwardLayer::operator=(const FeedForwardLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    std::copy(l._parameters->begin(), l._parameters->end(), _parameters->begin());

    return *this;
}

void FeedForwardLayer::initialize()
{
    auto initializationType = util::KnobDatabase::getKnobValue(
        "FeedForwardLayer::InitializationType", "glorot");

    if(initializationType == "glorot")
    {
        double e = util::KnobDatabase::getKnobValue(
            "FeedForwardLayer::RandomInitializationEpsilon", 6);

        double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));

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
            "FeedForwardLayer::RandomInitializationEpsilon", 1);

        double epsilon = std::sqrt((2.*e) / (getInputCount()));
        // generate normal random values with N(0,1)
        matrix::randn(_weights);

        // scale, the range is now [-epsilon, epsilon]
        apply(_weights, _weights, matrix::Multiply(epsilon));
    }

    // assign bias to 0.0
    apply(_bias, _bias, matrix::Fill(0.0));
}

void FeedForwardLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivations  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivations = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivations.size() == 1);

    auto inputActivation = foldTime(inputActivations.front());

    saveMatrix("inputActivation", inputActivation);

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << " Running forward propagation of matrix "
            << inputActivation.shapeString() << " through layer: "
            << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  input: " << inputActivation.debugString();
        util::log("FeedForwardLayer::Detail") << "  layer: " << _weights.debugString();
        util::log("FeedForwardLayer::Detail") << "  bias:  " << _bias.debugString();
    }
    else
    {
        util::log("FeedForwardLayer") << "  input shape: "
            << inputActivations.back().shapeString() << "\n";
    }

    auto unbiasedOutput = gemm(Matrix(_weights), false, 1.0, inputActivation, false);

    auto output = broadcast(unbiasedOutput, _bias, {}, matrix::Add());

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  output: " << output.debugString();
    }
    else
    {
        util::log("FeedForwardLayer") << "  output shape: " << output.shapeString() << "\n";
    }

    auto outputActivation = getActivationFunction()->apply(output);

    saveMatrix("outputActivation", outputActivation);

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  activation: "
            << outputActivation.debugString();
    }
    else
    {
        util::log("FeedForwardLayer") << "  activation: "
            << outputActivation.shapeString() << "\n";
    }

    outputActivations.push_back(unfoldTime(outputActivation,
        inputActivations.front().size()));
}

void FeedForwardLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients    = bundle[   "gradients"].get<MatrixVector>();
    auto& inputDeltas  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

    assert(outputDeltas.size() == 1);

    auto inputActivation  = loadMatrix("inputActivation" );
    auto outputActivation = loadMatrix("outputActivation");

    auto differenceWithTime = outputDeltas.front();
    auto difference = foldTime(differenceWithTime);

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << " Running reverse propagation on matrix ("
            << difference.size()[0] << " rows, " << difference.size()[1]
            << " columns) through layer with dimensions (" << getInputCount() << " inputs, "
            << getOutputCount() << " outputs).\n";
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
        util::log("FeedForwardLayer") << "  output size: " << outputActivation.shapeString()
            << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  output: " << outputActivation.debugString();
    }

    // finish computing the deltas
    auto deltas = apply(getActivationFunction()->applyDerivative(outputActivation),
        difference, matrix::Multiply());

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // compute gradient for the weights
    auto samples = differenceWithTime.size()[differenceWithTime.size().size() - 2];

    auto weightGradient = gemm(Matrix(deltas), false, 1.0 / samples, inputActivation, true);

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradient, weightGradient, getWeightCostFunction()->getGradient(_weights),
            matrix::Add());
    }

    gradients.push_back(std::move(weightGradient));

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer::Detail") << "  samples: " << samples << "\n";
        util::log("FeedForwardLayer") << "  weight grad shape: "
            << weightGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  weight grad: "
            << weightGradient.debugString();
    }

    // compute gradient for the bias
    auto biasGradient = reduce(apply(deltas, matrix::Divide(samples)), {1}, matrix::Add());

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  bias grad shape: "
            << biasGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  bias grad: "
            << biasGradient.debugString();
    }

    assert(biasGradient.size() == _bias.size());

    gradients.push_back(std::move(biasGradient));

    // compute deltas for previous layer
    auto deltasPropagatedReverse = gemm(_weights, true, deltas, false);

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

    if(util::isLogEnabled("FeedForwardLayer"))
    {
        util::log("FeedForwardLayer") << "  input deltas shape: "
            << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("FeedForwardLayer::Detail"))
    {
        util::log("FeedForwardLayer::Detail") << "  input deltas: "
            << previousLayerDeltas.debugString();
    }

    inputDeltas.push_back(unfoldTime(previousLayerDeltas, differenceWithTime.size()));
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

size_t FeedForwardLayer::getActivationMemory() const
{
    return precision().size() * getOutputCount();
}

void FeedForwardLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["weights"] = properties.path() + "." + properties.key() + ".weights.npy";
    properties["bias"]    = properties.path() + "." + properties.key() + ".bias.npy";

    saveToArchive(archive, properties["weights"], _weights);
    saveToArchive(archive, properties["bias"],    _bias);

    saveLayer(archive, properties);
}

void FeedForwardLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    _weights = matrix::loadFromArchive(archive, properties["weights"]);
    _bias    = matrix::loadFromArchive(archive, properties["bias"]);

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> FeedForwardLayer::clone() const
{
    return std::make_unique<FeedForwardLayer>(*this);
}

std::string FeedForwardLayer::getTypeName() const
{
    return "FeedForwardLayer";
}

}

}


