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
#include <lucius/network/interface/Bundle.h>

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
typedef matrix::IndexVector IndexVector;
typedef matrix::Dimension Dimension;

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer()
: BidirectionalRecurrentLayer(1, 1, 1, RECURRENT_FORWARD, RECURRENT_SIMPLE_TYPE,
    RECURRENT_LINEAR_INPUT)
{

}

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer(
    size_t layerSize, size_t expectedMiniBatchSize, size_t layers,
    int direction, int layerType, int inputMode)
: BidirectionalRecurrentLayer(layerSize, expectedMiniBatchSize, layers,
    direction, layerType, inputMode, matrix::SinglePrecision())
{

}

static RecurrentOpsHandle getHandle(size_t layerSize, size_t expectedMiniBatchSize, size_t layers,
    int direction, int layerType, int inputMode)
{
    return RecurrentOpsHandle(layerSize, expectedMiniBatchSize, layers,
        directions, layerType, inputMode);
}

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer(
    size_t layerSize, size_t expectedMiniBatchSize, size_t layers,
    int direction, int layerType, int inputMode,
    const matrix::Precision& precision)
: _parameters(new MatrixVector({createWeightsRecurrent(
  getHandle(layerSize, expectedMiniBatchSize, layers, direction, layerType, inputMode))})),
  _weights((*_parameters)[0]),
  _layerSize(layerSize),
  _expectedMiniBatchSize(expectedMiniBatchSize),
  _layers(layers),
  _direction(direction),
  _layerType(layerType),
  _inputMode(inputMode)
{

}

BidirectionalRecurrentLayer::~BidirectionalRecurrentLayer()
{

}

BidirectionalRecurrentLayer::BidirectionalRecurrentLayer(const BidirectionalRecurrentLayer& l)
: Layer(l),
  _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _weights((*_parameters)[0]),
  _layerSize(l._layerSize),
  _expectedMiniBatchSize(l._expectedMiniBatchSize),
  _layers(l._layers),
  _direction(l._direction),
  _layerType(l._layerType),
  _inputMode(l._inputMode)
{

}

BidirectionalRecurrentLayer& BidirectionalRecurrentLayer::operator=(
    const BidirectionalRecurrentLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _parameters = std::make_unique<MatrixVector>(*l._parameters);

    _layerSize  = l._layerSize;
    _layers     = l._layers;
    _direction  = l._direction;
    _layerType  = l._layerType;
    _inputMode  = l._inputMode;

    _expectedMiniBatchSize = l._expectedMiniBatchSize;

    return *this;
}

void BidirectionalRecurrentLayer::initialize()
{
    auto initializationType = util::KnobDatabase::getKnobValue(
        "BidirectionalRecurrentLayer::InitializationType", "glorot");

    auto handle = getHandle(_layerSize, _expectedMiniBatchSize, _layers,
        _direction, _layerType, _inputMode);

    for(size_t m = 0; m < getTotalWeightMatrices(); ++m)
    {
        auto slice = sliceLayerWeights(_weights, handle, m);

        if(isBiasLayer(handle, m))
        {
            zeros(slice);
        }
        else
        {
            if(initializationType == "glorot")
            {
                //Glorot
                double e = util::KnobDatabase::getKnobValue(
                    "BidirectionalRecurrentLayer::RandomInitializationEpsilon", 6);

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
                    "BidirectionalRecurrentLayer::RandomInitializationEpsilon", 1);

                double epsilon = std::sqrt((2.*e) / (getInputCount() * 2));

                // generate normal random values with N(0,1)
                matrix::randn(_weights);

                // scale, the range is now [-epsilon, epsilon]
                apply(_weights, _weights, matrix::Multiply(epsilon));
            }
        }
    }
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

void BidirectionalRecurrentLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<matrix::MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<matrix::MatrixVector>();

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
        util::log("BidirectionalRecurrentLayer::Detail") << "  input: "
            << inputActivations.debugString();
        util::log("BidirectionalRecurrentLayer::Detail") << "  weights: "
            << _weights.debugString();
    }

    size_t activationCount = inputActivations.size()[0];
    size_t miniBatch       = inputActivations.size()[inputActivations.size().size() - 2];
    size_t timesteps       = inputActivations.size().back();

    auto handle = getHandle(_layerSize, _expectedMiniBatchSize, _layers,
        _direction, _layerType, _inputMode);

    Matrix outputActivations({activationCount, miniBatch, timesteps}, precision());
    auto reserve = createReserveRecurrent(handle, precision());

    forwardPropRecurrent(outputActivations,
                         inputActivations,
                         reserve,
                         weights,
                         handle);

    saveMatrix("outputActivations", activation);

    if(bundle.contains("inputTimesteps"))
    {
        recurrentZeroEnds(reverseActivation, bundle["inputTimesteps"].get<IndexVector>());
    }

    saveMatrix("reserve", reserve);

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  activation: "
            << outputActivations.debugString();
    }
    else
    {
        util::log("BidirectionalRecurrentLayer") << "  activation: "
            << outputActivations.shapeString() << "\n";
    }

    outputActivationsVector.push_back(std::move(outputActivations));
}

void BidirectionalRecurrentLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients         = bundle[   "gradients"].get<matrix::MatrixVector>();
    auto& inputDeltaVector  = bundle[ "inputDeltas"].get<matrix::MatrixVector>();
    auto& outputDeltaVector = bundle["outputDeltas"].get<matrix::MatrixVector>();

    assert(outputDeltas.size() == 1);

    auto outputDeltas = unfoldTimeAndBatch(outputDeltaVector.back(), _expectedBatchSize);

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << " Running reverse propagation on matrix ("
            << outputDeltas.shapeString() << ") through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("BidirectionalRecurrentLayer") << "  layer: "
            << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  output deltas size: "
            << outputDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  output deltas: "
            << outputDeltas.debugString();
    }

    auto unfoldedOutputActivations = loadMatrix("outputActivations");
    auto reserve = loadMatrix("reserve");

    size_t activationCount = unfoldedOutputActivations.size()[0];
    size_t miniBatch = unfoldedOutputActivations.size()[
        unfoldedOutputActivations.size().size() - 2];
    size_t timesteps = unfoldedOutputActivations.size().back();

    Matrix inputDeltas({activationCount, miniBatch, timesteps}, precision());

    backPropDeltasRecurrent(inputDeltas,
                            outputDeltas,
                            weights,
                            unfoldedOutputActivations,
                            reserve,
                            handle);

    auto unfoldedInputActivations = loadMatrix("inputActivations");

    Matrix weightGradients(_weights.size(), precision());

    backPropGradientsRecurrent(weightGradients,
                               unfoldedInputActivations,
                               unfoldedOutputActivations,
                               outputDeltas,
                               reserve,
                               handle);

    // Compute recurrent weight gradients
    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  input deltas size: "
            << inputDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  input deltas: "
            << inputDeltas.debugString();
    }

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradients, weightGradients,
            getWeightCostFunction()->getGradient(_weights), matrix::Add());
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer"))
    {
        util::log("BidirectionalRecurrentLayer") << "  weight grad shape: "
            << weightGradients.shapeString() << "\n";
    }

    if(util::isLogEnabled("BidirectionalRecurrentLayer::Detail"))
    {
        util::log("BidirectionalRecurrentLayer::Detail") << "  weight grad: "
            << weightGradients.debugString();
    }

    gradients.push_back(std::move(weightGradients));

    inputDeltaVector.push_back(reshape(inputDeltas, {getInputCount(), miniBatch, timesteps}));
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
    return _weights.precision();
}

double BidirectionalRecurrentLayer::computeWeightCost() const
{
    return getWeightCostFunction()->getCost(_weights);
}

Dimension BidirectionalRecurrentLayer::getInputSize() const
{
    return {_layerSize, 1, 1};
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
    return getInputCount() * _layers;
}

size_t BidirectionalRecurrentLayer::totalConnections() const
{
    return _weights.elements();
}

size_t BidirectionalRecurrentLayer::getFloatingPointOperationCount() const
{
    return 2 * totalConnections();
}

size_t BidirectionalRecurrentLayer::getActivationMemory() const
{
    return getOutputCount() * precision().size();
}

void BidirectionalRecurrentLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["weights"] = properties.path() + "." + properties.key() + ".weights.npy";

    properties["layer-size"] = std::to_string(_layerSize);
    properties["batch-size"] = std::to_string(_expectedMiniBatchSize);
    properties["layers"]     = std::to_string(_layers);

    properties["direction"]  = std::to_string(_direction);
    properties["layer-type"] = std::to_string(_layerType);
    properties["input-mode"] = std::to_string(_inputMode);

    saveToArchive(archive, properties["weights"], _weights);

    saveLayer(archive, properties);
}

void BidirectionalRecurrentLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _weights = matrix::loadFromArchive(archive, properties["weights"]);

    _layerSize         = properties.get<size_t>("layer-size");
    _expectedBatchSize = properties.get<size_t>("batch-size");
    _layers            = properties.get<size_t>("layers");

    _direction = properties.get<int>("direction");
    _layerType = properties.get<int>("layer-type");
    _inputMode = properties.get<int>("input-mode");

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



