/*  \file   RecurrentLayer.cpp
    \author Gregory Diamos
     \date   Dec 24, 2014
     \brief  The implementation of the RecurrentLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/RecurrentLayer.h>

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

RecurrentLayer::RecurrentLayer()
: RecurrentLayer(1, 1)
{

}

RecurrentLayer::RecurrentLayer(size_t size, size_t batchSize)
: RecurrentLayer(size, batchSize, matrix::SinglePrecision())
{

}

RecurrentLayer::RecurrentLayer(size_t size, size_t batchSize, const matrix::Precision& precision)
: _parameters(new MatrixVector({Matrix({size, size}, precision),
    Matrix({size}, precision), Matrix({size, size}, precision)})),
  _forwardWeights((*_parameters)[0]), _bias((*_parameters)[1]),
  _recurrentWeights((*_parameters)[2]), _expectedBatchSize(batchSize)
{

}

RecurrentLayer::~RecurrentLayer()
{

}

RecurrentLayer::RecurrentLayer(const RecurrentLayer& l)
: _parameters(std::make_unique<MatrixVector>(*l._parameters)),
 _forwardWeights((*_parameters)[0]),
 _bias((*_parameters)[1]),
 _recurrentWeights((*_parameters)[2]),
 _expectedBatchSize(l._expectedBatchSize)
{

}

RecurrentLayer& RecurrentLayer::operator=(const RecurrentLayer& l)
{
    if(&l == this)
    {
        return *this;
    }

    _parameters = std::move(std::make_unique<MatrixVector>(*l._parameters));
    _expectedBatchSize = l._expectedBatchSize;

    return *this;
}

void RecurrentLayer::initialize()
{
    double e = util::KnobDatabase::getKnobValue("Layer::RandomInitializationEpsilon", 6);

    double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));

    // generate uniform random values between [0, 1]
    matrix::rand(_forwardWeights);
    matrix::rand(_recurrentWeights);

    // shift to center on 0, the range is now [-0.5, 0.5]
    apply(_forwardWeights, _forwardWeights, matrix::Add(-0.5));
    apply(_recurrentWeights, _recurrentWeights, matrix::Add(-0.5));

    // scale, the range is now [-epsilon, epsilon]
    apply(_forwardWeights, _forwardWeights, matrix::Multiply(2.0 * epsilon));
    apply(_recurrentWeights, _recurrentWeights, matrix::Multiply(2.0 * epsilon));

    // assign bias to 0.0f
    apply(_bias, _bias, matrix::Fill(0.0f));
}

static matrix::Matrix unfoldTimeAndBatch(const Matrix& input, size_t batchSize)
{
    size_t activationCount = input.size()[0];
    size_t miniBatch       = 1;
    size_t timesteps       = 1;

    if(input.size().size() == 2)
    {
        miniBatch = batchSize;
        timesteps = input.size()[1] / miniBatch;
    }
    else
    {
        miniBatch = input.size()[1];
        timesteps = input.size()[2];
    }

    return reshape(input, {activationCount, miniBatch, timesteps});
}

void RecurrentLayer::runForwardImplementation(MatrixVector& activations)
{
    auto inputActivations = unfoldTimeAndBatch(activations.back(), _expectedBatchSize);

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << " Running forward propagation through layer: " << _forwardWeights.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  input: " << inputActivations.debugString();
        util::log("RecurrentLayer::Detail") << "  forward-weights: " << _forwardWeights.debugString();
        util::log("RecurrentLayer::Detail") << "  recurrent-weights: " << _recurrentWeights.debugString();
        util::log("RecurrentLayer::Detail") << "  bias:  " << _bias.debugString();
    }

    size_t activationCount = inputActivations.size()[0];
    size_t miniBatch       = inputActivations.size()[1];
    size_t timesteps       = inputActivations.size()[2];

    auto unbiasedOutput = gemm(Matrix(_forwardWeights), false, 1.0, reshape(inputActivations, {activationCount, miniBatch * timesteps}), false);

    auto activation = broadcast(unbiasedOutput, _bias, {}, matrix::Add());

    activations.push_back(copy(activation));

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  before-recurrent activation: " << activation.debugString();
    }
    else
    {
        util::log("RecurrentLayer") << "  before-recurrent activation: " << activation.shapeString() << "\n";
    }

    activation = reshape(activation, {activationCount, miniBatch, timesteps});

    forwardRecurrentActivations(activation, _recurrentWeights, matrix::RECURRENT_FORWARD_TIME, getActivationFunction()->getOperation());

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  activation: " << activation.debugString();
    }
    else
    {
        util::log("RecurrentLayer") << "  activation: " << activation.shapeString() << "\n";
    }

    activations.push_back(activation);
}

Matrix RecurrentLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& activations,
    const Matrix& foldedDeltas)
{
    auto deltas = unfoldTimeAndBatch(foldedDeltas, _expectedBatchSize);

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << " Running reverse propagation on matrix (" << deltas.shapeString()
            << ") through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("RecurrentLayer") << "  layer: " << _forwardWeights.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  deltas size: " << deltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  deltas: " << deltas.debugString();
    }

    // Compute deltas for intermediate time steps and the feed forward activations
    auto outputActivations = unfoldTimeAndBatch(activations.back(), _expectedBatchSize);
    activations.pop_back();

    auto forwardDeltas = reverseRecurrentDeltas(Matrix(deltas), _recurrentWeights, outputActivations, matrix::RECURRENT_FORWARD_TIME,
        getActivationFunction()->getDerivativeOperation());

    // Compute recurrent weight gradients
    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  forward deltas size: " << forwardDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  forward deltas: " << forwardDeltas.debugString();
    }

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  output size: " << outputActivations.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  output: " << outputActivations.debugString();
    }

    auto recurrentWeightGradients = reverseRecurrentGradients(outputActivations, forwardDeltas, matrix::RECURRENT_FORWARD_TIME);

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  recurrent weight grad shape: " << recurrentWeightGradients.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  recurrent weight grad: " << recurrentWeightGradients.debugString();
    }

    auto forwardOutputActivations = activations.back();
    activations.pop_back();

    size_t activationCount = activations.back().size()[0];
    size_t miniBatch       = activations.back().size()[1];
    size_t timesteps       = activations.back().size()[2];

    auto forwardInputActivations = reshape(activations.back(), {activationCount, miniBatch * timesteps});

    forwardDeltas = reshape(forwardDeltas, {activationCount, miniBatch * timesteps});

    // compute gradient for the forward weights
    auto samples = miniBatch;

    auto weightGradient = gemm(Matrix(forwardDeltas), false, 1.0 / samples, forwardInputActivations, true);

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradient, weightGradient, getWeightCostFunction()->getGradient(_forwardWeights), matrix::Add());
    }

    gradients.push_back(std::move(weightGradient));

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  forward weight grad shape: " << weightGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  forward weight grad: " << weightGradient.debugString();
    }

    // compute gradient for the bias
    auto biasGradient = reduce(apply(forwardDeltas, matrix::Divide(samples)), {1}, matrix::Add());

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  bias grad shape: " << biasGradient.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  bias grad: " << biasGradient.debugString();
    }

    assert(biasGradient.size() == _bias.size());

    gradients.push_back(std::move(biasGradient));

    gradients.push_back(std::move(recurrentWeightGradients));

    // compute deltas for previous layer
    auto deltasPropagatedReverse = gemm(_forwardWeights, true, forwardDeltas, false);

    Matrix previousLayerDeltas;

    if(getActivationCostFunction() != nullptr)
    {
        auto activationCostFunctionGradient = getActivationCostFunction()->getGradient(forwardOutputActivations);

        apply(previousLayerDeltas, deltasPropagatedReverse, activationCostFunctionGradient, matrix::Multiply());
    }
    else
    {
        previousLayerDeltas = std::move(deltasPropagatedReverse);
    }

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  output shape: " << previousLayerDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  output: " << previousLayerDeltas.debugString();
    }

    return reshape(previousLayerDeltas, {getInputCount(), miniBatch, timesteps});
}

MatrixVector& RecurrentLayer::weights()
{
    return *_parameters;
}

const MatrixVector& RecurrentLayer::weights() const
{
    return *_parameters;
}

const matrix::Precision& RecurrentLayer::precision() const
{
    return _forwardWeights.precision();
}

double RecurrentLayer::computeWeightCost() const
{
    return getWeightCostFunction()->getCost(_forwardWeights) +
        getWeightCostFunction()->getCost(_recurrentWeights);
}

Dimension RecurrentLayer::getInputSize() const
{
    return {_forwardWeights.size()[1], 1, 1};
}

Dimension RecurrentLayer::getOutputSize() const
{
    return getInputSize();
}

size_t RecurrentLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t RecurrentLayer::getOutputCount() const
{
    return getInputCount();
}

size_t RecurrentLayer::totalNeurons() const
{
    return 2 * getInputCount();
}

size_t RecurrentLayer::totalConnections() const
{
    return 2 * _forwardWeights.elements() + _bias.elements();
}

size_t RecurrentLayer::getFloatingPointOperationCount() const
{
    return 2 * totalConnections();
}

void RecurrentLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["forward-weights"]   = properties.path() + "." + properties.key() + ".forward-weights.npy";
    properties["recurrent-weights"] = properties.path() + "." + properties.key() + ".recurrent-weights.npy";
    properties["bias"]              = properties.path() + "." + properties.key() + ".bias.npy";

    properties["batch-size"] = std::to_string(_expectedBatchSize);

    saveToArchive(archive, properties["forward-weights"],   _forwardWeights);
    saveToArchive(archive, properties["recurrent-weights"], _recurrentWeights);
    saveToArchive(archive, properties["bias"],    _bias);

    saveLayer(archive, properties);
}

void RecurrentLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    _forwardWeights   = matrix::loadFromArchive(archive, properties["forward-weights"]);
    _recurrentWeights = matrix::loadFromArchive(archive, properties["recurrent-weights"]);
    _bias             = matrix::loadFromArchive(archive, properties["bias"]);

    _expectedBatchSize = properties.get<size_t>("batch-size");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> RecurrentLayer::clone() const
{
    return std::make_unique<RecurrentLayer>(*this);
}

std::unique_ptr<Layer> RecurrentLayer::mirror() const
{
    return clone();
}

std::string RecurrentLayer::getTypeName() const
{
    return "RecurrentLayer";
}

}

}


