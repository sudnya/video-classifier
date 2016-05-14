/*  \file   BidirectionalRecurrentLayer.cpp
    \author Sudnya Diamos
     \date   May 9, 2016
     \brief  The implementation of the BidirectionalRecurrentLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/BidirectionalRecurrentLayer.h>

#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/ActivationCostFunction.h>
#include <lucius/network/interface/WeightCostFunction.h>

#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/RandomOperations.h>
#include <lucius/matrix/interface/RecurrentOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/FileOperations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/TarArchive.h>
#include <lucius/util/interface/PropertyTree.h>

namespace lucius
{

namespace network
{

typedef matrix::Matrix Matrix;
typedef matrix::MatrixVector MatrixVector;
typedef matrix::Dimension Dimension;

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer()
: BidirectionalRecurrentLayer(1, 1)
{

}

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer(size_t size, size_t batchSize)
: BidirectionalRecurrentLayer(size, batchSize, matrix::SinglePrecision())
{

}

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer(size_t size, size_t batchSize, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({size, size}, precision),
    Matrix({size}, precision), Matrix({size, size}, precision), Matrix({size, size}, precision)})),
  _forwardWeights((*_parameters)[0]), _bias((*_parameters)[1]),
  _recurrentForwardWeights((*_parameters)[2]), _recurrentReverseWeights((*_parameters)[3]), _expectedBatchSize(batchSize)
{

}

BidirectionalRecurrentLayer::~BidirectionalRecurrentLayer()
{

}

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer(const BidirectionalRecurrentLayer& l)
: Layer(l), _parameters(std::make_unique<MatrixVector>(*l._parameters)),
 _forwardWeights((*_parameters)[0]),
 _bias((*_parameters)[1]),
 _recurrentForwardWeights((*_parameters)[2]),
 _recurrentReverseWeights((*_parameters)[3]),
 _expectedBatchSize(l._expectedBatchSize)
{

}

BidirectionalRecurrentLayer& BidirectionalRecurrentLayer::operator=(const BidirectionalRecurrentLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _parameters = std::make_unique<MatrixVector>(*l._parameters);
    _expectedBatchSize = l._expectedBatchSize;

    return *this;
}

void BidirectionalRecurrentLayer::initialize()
{
    auto initializationType = util::KnobDatabase::getKnobValue(
        "BidirectionalRecurrentLayer::InitializationType", "glorot");

    if(initializationType == "glorot")
    {
        //Glorot
        double e = util::KnobDatabase::getKnobValue(
            "BidirectionalRecurrentLayer::RandomInitializationEpsilon", 6);

        double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));

        // generate uniform random values between [0, 1]
        matrix::rand(_forwardWeights);
        matrix::rand(_recurrentForwardWeights);
        matrix::rand(_recurrentReverseWeights);

        // shift to center on 0, the range is now [-0.5, 0.5]
        apply(_forwardWeights, _forwardWeights, matrix::Add(-0.5));
        apply(_recurrentForwardWeights, _recurrentForwardWeights, matrix::Add(-0.5));
        apply(_recurrentReverseWeights, _recurrentReverseWeights, matrix::Add(-0.5));

        // scale, the range is now [-epsilon, epsilon]
        apply(_forwardWeights, _forwardWeights, matrix::Multiply(2.0 * epsilon));
        apply(_recurrentForwardWeights, _recurrentForwardWeights, matrix::Multiply(2.0 * epsilon));
        apply(_recurrentReverseWeights, _recurrentReverseWeights, matrix::Multiply(2.0 * epsilon));
    }
    else if(initializationType == "he")
    {
        // He
        double e = util::KnobDatabase::getKnobValue(
            "BidirectionalRecurrentLayer::RandomInitializationEpsilon", 1);

        double epsilon = std::sqrt((2.*e) / (getInputCount() * 2));

        // generate normal random values with N(0,1)
        matrix::randn(_forwardWeights);
        matrix::randn(_recurrentForwardWeights);
        matrix::randn(_recurrentReverseWeights);

        // scale, the range is now [-epsilon, epsilon]
        apply(_forwardWeights, _forwardWeights, matrix::Multiply(epsilon));
        apply(_recurrentForwardWeights, _recurrentForwardWeights, matrix::Multiply(epsilon));
        apply(_recurrentReverseWeights, _recurrentReverseWeights, matrix::Multiply(epsilon));
    }

    // assign bias to 0.0f
    apply(_bias, _bias, matrix::Fill(0.0f));
}

static matrix::Matrix unfoldTimeAndBatch(const Matrix& input, size_t batchSize)
{
    size_t activationCount = input.size()[0];

    size_t limit = input.size().size() - 2;

    for(size_t i = 1; i < limit; ++i)
    {
        activationCount *= input.size()[i];
    }

    size_t miniBatch = 1;
    size_t timesteps = 1;

    if(input.size().size() == 2)
    {
        miniBatch = batchSize;
        timesteps = input.size()[1] / miniBatch;
    }
    else
    {
        miniBatch = input.size()[input.size().size() - 2];
        timesteps = input.size()[input.size().size() - 1];
    }

    return reshape(input, {activationCount, miniBatch, timesteps});
}

void BidirectionalRecurrentLayer::runForwardImplementation(MatrixVector& outputActivationsVector,
    const MatrixVector& inputActivationsVector)
{
    assert(inputActivationsVector.size() == 1);

    saveMatrix("inputActivations", inputActivationsVector.back());

    auto inputActivations = unfoldTimeAndBatch(inputActivationsVector.back(), _expectedBatchSize);

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << " Running forward propagation through layer: "
            << _forwardWeights.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  input: " << inputActivations.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  forward-weights: "
            << _forwardWeights.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  recurrent-forward-weights: "
            << _recurrentForwardWeights.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  recurrent-reverse-weights: "
            << _recurrentReverseWeights.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  bias:  " << _bias.debugString();
    }

    size_t activationCount = inputActivations.size()[0];
    size_t miniBatch       = inputActivations.size()[inputActivations.size().size() - 2];
    size_t timesteps       = inputActivations.size().back();

    auto unbiasedOutput = gemm(Matrix(_forwardWeights), false, 1.0, reshape(inputActivations,
        {activationCount, miniBatch * timesteps}), false);

    auto activation = broadcast(unbiasedOutput, _bias, {}, matrix::Add());

    saveMatrix("forwardOutputActivations", copy(activation));

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  before-recurrent activation: "
            << activation.debugString();
    }
    else
    {
        util::log("BidirectionalRecurrentLayer") << "  before-recurrent activation: "
            << activation.shapeString() << "\n";
    }

    activation = reshape(activation, {activationCount, miniBatch, timesteps});

    auto reverseActivation = copy(activation);

    forwardRecurrentActivations(activation, _recurrentForwardWeights,
        lucius::matrix::RECURRENT_FORWARD_TIME,
        getActivationFunction()->getOperation());

    saveMatrix("forwardOutputActivations", activation);

    forwardRecurrentActivations(reverseActivation, _recurrentReverseWeights,
        lucius::matrix::RECURRENT_REVERSE_TIME,
        getActivationFunction()->getOperation());

    saveMatrix("reverseOutputActivations", reverseActivation);

    auto output = apply(Matrix(activation), reverseActivation, matrix::Add());

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  activation: " << activation.debugString();
    }
    else
    {
        util::log("BidirectionalRecurrentLayer") << "  activation: " << activation.shapeString() << "\n";
    }

    outputActivationsVector.push_back(std::move(output));
}

void BidirectionalRecurrentLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& inputDeltas,
    const MatrixVector& outputDeltas)
{
    assert(outputDeltas.size() == 1);

    auto deltas = unfoldTimeAndBatch(outputDeltas.back(), _expectedBatchSize);

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << " Running reverse propagation on matrix ("
            << deltas.shapeString() << ") through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("BidirectionalRecurrentLayer") << "  layer: " << _forwardWeights.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  deltas size: " << deltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // Compute deltas for intermediate time steps and the feed forward activations
    auto reverseTimeOutputActivations = unfoldTimeAndBatch(loadMatrix("reverseOutputActivations"),
        _expectedBatchSize);
    auto forwardTimeOutputActivations = unfoldTimeAndBatch(loadMatrix("forwardOutputActivations"),
        _expectedBatchSize);

    auto recurrentForwardDeltas = reverseRecurrentDeltas(Matrix(deltas), _recurrentForwardWeights,
        forwardTimeOutputActivations, matrix::RECURRENT_FORWARD_TIME,
        getActivationFunction()->getDerivativeOperation());

    auto recurrentReverseDeltas = reverseRecurrentDeltas(Matrix(deltas), _recurrentReverseWeights,
        reverseTimeOutputActivations, matrix::RECURRENT_REVERSE_TIME,
        getActivationFunction()->getDerivativeOperation());

    auto forwardDeltas = apply(Matrix(recurrentForwardDeltas), recurrentReverseDeltas, matrix::Add());

    // Compute recurrent weight gradients
    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  forward deltas size: "
            << forwardDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  forward deltas: "
            << forwardDeltas.debugString();
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  output forward size: "
            << forwardTimeOutputActivations.shapeString() << "\n";
        util::log("BidirectionalRecurrentLayer") << "  output reverse size: "
            << reverseTimeOutputActivations.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  output forward: "
            << forwardTimeOutputActivations.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  output reverse: "
            << reverseTimeOutputActivations.debugString();
    }

    auto recurrentForwardWeightGradients = reverseRecurrentGradients(forwardTimeOutputActivations,
        recurrentForwardDeltas, lucius::matrix::RECURRENT_FORWARD_TIME);
    auto recurrentReverseWeightGradients = reverseRecurrentGradients(reverseTimeOutputActivations,
        recurrentReverseDeltas, lucius::matrix::RECURRENT_REVERSE_TIME);

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  recurrent forward weight grad shape: "
            << recurrentForwardWeightGradients.shapeString() << "\n";
        util::log("BidirectionalRecurrentLayer") << "  recurrent reverse weight grad shape: "
            << recurrentReverseWeightGradients.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  recurrent forward weight grad: "
            << recurrentForwardWeightGradients.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  recurrent reverse weight grad: "
            << recurrentReverseWeightGradients.debugString();
    }

    auto forwardOutputActivations = loadMatrix("forwardOutputActivations");

    auto unfoldedInputActivations = loadMatrix("inputActivations");

    size_t activationCount = unfoldedInputActivations.size()[0];
    size_t miniBatch       = unfoldedInputActivations.size()[unfoldedInputActivations.size().size() - 2];
    size_t timesteps       = unfoldedInputActivations.size().back();

    auto forwardInputActivations = reshape(unfoldedInputActivations,
        {activationCount, miniBatch * timesteps});

    forwardDeltas = reshape(forwardDeltas, {activationCount, miniBatch * timesteps});

    // compute gradient for the forward weights
    auto samples = miniBatch;

    auto weightGradient = gemm(Matrix(forwardDeltas), false, 1.0 / samples,
        forwardInputActivations, true);

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradient, weightGradient,
            getWeightCostFunction()->getGradient(_forwardWeights), matrix::Add());
    }

    gradients.push_back(std::move(weightGradient));

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  forward weight grad shape: "
            << weightGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  forward weight grad: "
            << weightGradient.debugString();
    }

    // compute gradient for the bias
    auto biasGradient = reduce(apply(forwardDeltas, matrix::Divide(samples)),
        {1}, matrix::Add());

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  bias grad shape: " << biasGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  bias grad: " << biasGradient.debugString();
    }

    assert(biasGradient.size() == _bias.size());

    gradients.push_back(std::move(biasGradient));

    gradients.push_back(std::move(recurrentForwardWeightGradients));

    gradients.push_back(std::move(recurrentReverseWeightGradients));

    // compute deltas for previous layer
    auto deltasPropagatedReverse = gemm(_forwardWeights, true, forwardDeltas, false);

    Matrix previousLayerDeltas;

    if(getActivationCostFunction() != nullptr)
    {
        auto activationCostFunctionGradient =
            getActivationCostFunction()->getGradient(forwardOutputActivations);

        apply(previousLayerDeltas, deltasPropagatedReverse,
            activationCostFunctionGradient, matrix::Multiply());
    }
    else
    {
        previousLayerDeltas = std::move(deltasPropagatedReverse);
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  output shape: "
            << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  output: " << previousLayerDeltas.debugString();
    }

    inputDeltas.push_back(reshape(previousLayerDeltas, {getInputCount(), miniBatch, timesteps}));
}

MatrixVector& BidirectionalRecurrentLayer::weights()
{
    return *_parameters;
}

const MatrixVector& BidirectionalRecurrentLayer::weights() const
{
    return *_parameters;
}

const matrix::Precision& BidirectionalRecurrentLayer::precision() const
{
    return _forwardWeights.precision();
}

double BidirectionalRecurrentLayer::computeWeightCost() const
{
    return getWeightCostFunction()->getCost(_forwardWeights) +
        getWeightCostFunction()->getCost(_recurrentForwardWeights) +
        getWeightCostFunction()->getCost(_recurrentReverseWeights);
}

Dimension BidirectionalRecurrentLayer::getInputSize() const
{
    return {_forwardWeights.size()[1], 1, 1};
}

Dimension BidirectionalRecurrentLayer::getOutputSize() const
{
    return getInputSize();
}

size_t BidirectionalRecurrentLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t BidirectionalRecurrentLayer::getOutputCount() const
{
    return getInputCount();
}

size_t BidirectionalRecurrentLayer::totalNeurons() const
{
    return 3 * getInputCount();
}

size_t BidirectionalRecurrentLayer::totalConnections() const
{
    return 3 * _forwardWeights.elements() + _bias.elements();
}

size_t BidirectionalRecurrentLayer::getFloatingPointOperationCount() const
{
    return 3 * totalConnections();
}

size_t BidirectionalRecurrentLayer::getActivationMemory() const
{
    return getOutputCount() * precision().size();
}

void BidirectionalRecurrentLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["forward-weights"] =
        properties.path() + "." + properties.key() + ".forward-weights.npy";
    properties["recurrent-forward-weights"] =
        properties.path() + "." + properties.key() + ".recurrent-forward-weights.npy";
    properties["recurrent-reverse-weights"] =
        properties.path() + "." + properties.key() + ".recurrent-reverse-weights.npy";

    properties["bias"] = properties.path() + "." + properties.key() + ".bias.npy";

    properties["batch-size"] = std::to_string(_expectedBatchSize);

    saveToArchive(archive, properties["forward-weights"],   _forwardWeights);
    saveToArchive(archive, properties["recurrent-forward-weights"], _recurrentForwardWeights);
    saveToArchive(archive, properties["recurrent-reverse-weights"], _recurrentReverseWeights);
    saveToArchive(archive, properties["bias"],    _bias);

    saveLayer(archive, properties);
}

void BidirectionalRecurrentLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    _forwardWeights          = matrix::loadFromArchive(archive, properties["forward-weights"]);
    _recurrentForwardWeights = matrix::loadFromArchive(archive, properties["recurrent-forward-weights"]);
    _recurrentReverseWeights = matrix::loadFromArchive(archive, properties["recurrent-reverse-weights"]);
    _bias                    = matrix::loadFromArchive(archive, properties["bias"]);

    _expectedBatchSize = properties.get<size_t>("batch-size");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> BidirectionalRecurrentLayer::clone() const
{
    return std::make_unique<BidirectionalRecurrentLayer>(*this);
}

std::string BidirectionalRecurrentLayer::getTypeName() const
{
    return "BidirectionalRecurrentLayer";
}

}

}



