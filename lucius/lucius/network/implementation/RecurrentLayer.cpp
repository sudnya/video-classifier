/*  \file   RecurrentLayer.cpp
    \author Sudnya Diamos
    \date   May 9, 2016
    \brief  The implementation of the RecurrentLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/RecurrentLayer.h>

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

RecurrentLayer::RecurrentLayer()
: RecurrentLayer(1, 1, matrix::RECURRENT_FORWARD, matrix::RECURRENT_SIMPLE_TYPE,
    matrix::RECURRENT_LINEAR_INPUT)
{

}

RecurrentLayer::RecurrentLayer(
    size_t layerSize, size_t layers,
    int direction, int layerType, int inputMode)
: RecurrentLayer(layerSize, layers,
    direction, layerType, inputMode, matrix::Precision::getDefaultPrecision())
{

}

static matrix::RecurrentOpsHandle getHandle(size_t layerSize, size_t expectedMiniBatchSize,
    size_t timesteps, size_t layers, int direction, int layerType, int inputMode)
{
    return matrix::RecurrentOpsHandle(layerSize, expectedMiniBatchSize, timesteps, layers,
        static_cast<matrix::RecurrentLayerDirection>(direction),
        static_cast<matrix::RecurrentLayerType>(layerType),
        static_cast<matrix::RecurrentLayerInputMode>(inputMode));
}

RecurrentLayer::RecurrentLayer(
    size_t layerSize, size_t layers,
    int direction, int layerType, int inputMode,
    const matrix::Precision& precision)
: _parameters(new MatrixVector({createWeightsRecurrent(
    getHandle(layerSize, 1, 1, layers, direction, layerType, inputMode),
    precision)})),
  _weights((*_parameters)[0]),
  _layerSize(layerSize),
  _layers(layers),
  _direction(direction),
  _layerType(layerType),
  _inputMode(inputMode)
{

}

RecurrentLayer::~RecurrentLayer()
{

}

RecurrentLayer::RecurrentLayer(const RecurrentLayer& l)
: Layer(l),
  _parameters(std::make_unique<MatrixVector>(*l._parameters)),
  _weights((*_parameters)[0]),
  _layerSize(l._layerSize),
  _layers(l._layers),
  _direction(l._direction),
  _layerType(l._layerType),
  _inputMode(l._inputMode)
{

}

RecurrentLayer& RecurrentLayer::operator=(
    const RecurrentLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    std::copy(l._parameters->begin(), l._parameters->end(), _parameters->begin());

    _layerSize  = l._layerSize;
    _layers     = l._layers;
    _direction  = l._direction;
    _layerType  = l._layerType;
    _inputMode  = l._inputMode;

    return *this;
}

void RecurrentLayer::initialize()
{
    auto initializationType = util::KnobDatabase::getKnobValue(
        "RecurrentLayer::InitializationType", "glorot");

    auto handle = getHandle(_layerSize, 1, 1, _layers,
        _direction, _layerType, _inputMode);

    for(size_t m = 0; m < getTotalWeightMatrices(handle); ++m)
    {
        auto weightSlice = sliceLayerWeights(_weights, handle, m);

        if(isBiasMatrix(handle, m))
        {
            zeros(weightSlice);
        }
        else
        {
            if(initializationType == "glorot")
            {
                //Glorot
                double e = util::KnobDatabase::getKnobValue(
                    "RecurrentLayer::RandomInitializationEpsilon", 6);

                double epsilon = std::sqrt((e) / (getInputCount() + getOutputCount() + 1));

                // generate uniform random values between [0, 1]
                matrix::rand(weightSlice);

                // shift to center on 0, the range is now [-0.5, 0.5]
                apply(weightSlice, weightSlice, matrix::Add(-0.5));

                // scale, the range is now [-epsilon, epsilon]
                apply(weightSlice, weightSlice, matrix::Multiply(2.0 * epsilon));
            }
            else if(initializationType == "he")
            {
                // He
                double e = util::KnobDatabase::getKnobValue(
                    "RecurrentLayer::RandomInitializationEpsilon", 1);

                double epsilon = std::sqrt((2.*e) / (getInputCount() * 2));

                // generate normal random values with N(0,1)
                matrix::randn(weightSlice);

                // scale, the range is now [-epsilon, epsilon]
                apply(weightSlice, weightSlice, matrix::Multiply(epsilon));
            }
        }
    }
}

void RecurrentLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<matrix::MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<matrix::MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    saveMatrix("inputActivations", inputActivations);

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << " Running forward propagation through layer: "
            << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  input: "
            << inputActivations.debugString();
        util::log("RecurrentLayer::Detail") << "  weights: "
            << _weights.debugString();
    }

    size_t activationCount = inputActivations.size()[0];
    size_t miniBatch       = inputActivations.size()[inputActivations.size().size() - 2];
    size_t timesteps       = inputActivations.size().back();

    auto handle = getHandle(_layerSize, miniBatch, timesteps, _layers,
        _direction, _layerType, _inputMode);

    size_t outputActivationCount = activationCount;

    if(_direction == matrix::RECURRENT_BIDIRECTIONAL)
    {
        outputActivationCount *= 2;
    }

    Matrix outputActivations({outputActivationCount, miniBatch, timesteps}, precision());
    auto reserve = createReserveRecurrent(handle, precision());

    if(bundle.contains("inputTimesteps") && _direction == matrix::RECURRENT_REVERSE)
    {
        recurrentZeroEnds(inputActivations, bundle["inputTimesteps"].get<IndexVector>());
    }

    forwardPropRecurrent(outputActivations,
                         inputActivations,
                         reserve,
                         _weights,
                         handle);

    saveMatrix("outputActivations", outputActivations);

    saveMatrix("reserve", reserve);

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  activation: "
            << outputActivations.debugString();
    }
    else
    {
        util::log("RecurrentLayer") << "  activation: "
            << outputActivations.shapeString() << "\n";
    }

    outputActivationsVector.push_back(std::move(outputActivations));
}

void RecurrentLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients         = bundle[   "gradients"].get<matrix::MatrixVector>();
    auto& inputDeltaVector  = bundle[ "inputDeltas"].get<matrix::MatrixVector>();
    auto& outputDeltaVector = bundle["outputDeltas"].get<matrix::MatrixVector>();

    assert(outputDeltaVector.size() == 1);

    auto outputDeltas = outputDeltaVector.back();

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << " Running reverse propagation on matrix ("
            << outputDeltas.shapeString() << ") through layer with dimensions ("
            << getInputCount() << " inputs, " << getOutputCount() << " outputs).\n";
        util::log("RecurrentLayer") << "  layer: "
            << _weights.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  output deltas size: "
            << outputDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  output deltas: "
            << outputDeltas.debugString();
    }

    auto unfoldedOutputActivations = loadMatrix("outputActivations");
    auto reserve = loadMatrix("reserve");

    size_t activationCount = unfoldedOutputActivations.size()[0];
    size_t miniBatch = unfoldedOutputActivations.size()[
        unfoldedOutputActivations.size().size() - 2];
    size_t timesteps = unfoldedOutputActivations.size().back();

    auto handle = getHandle(_layerSize, miniBatch, timesteps, _layers,
        _direction, _layerType, _inputMode);

    size_t inputActivationCount = activationCount;

    if(_direction == matrix::RECURRENT_BIDIRECTIONAL)
    {
        inputActivationCount /= 2;
    }

    Matrix inputDeltas({inputActivationCount, miniBatch, timesteps}, precision());

    backPropDeltasRecurrent(inputDeltas,
                            outputDeltas,
                            _weights,
                            unfoldedOutputActivations,
                            reserve,
                            handle);

    auto unfoldedInputActivations = loadMatrix("inputActivations");

    Matrix weightGradients = zeros(_weights.size(), precision());

    backPropGradientsRecurrent(weightGradients,
                               unfoldedInputActivations,
                               unfoldedOutputActivations,
                               reserve,
                               handle);

    apply(weightGradients, weightGradients, matrix::Divide(miniBatch));

    // Compute recurrent weight gradients
    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  input deltas size: "
            << inputDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  input deltas: "
            << inputDeltas.debugString();
    }

    // add in the weight cost function term
    if(getWeightCostFunction() != nullptr)
    {
        apply(weightGradients, weightGradients,
            getWeightCostFunction()->getGradient(_weights), matrix::Add());
    }

    if(util::isLogEnabled("RecurrentLayer"))
    {
        util::log("RecurrentLayer") << "  weight grad shape: "
            << weightGradients.shapeString() << "\n";
    }

    if(util::isLogEnabled("RecurrentLayer::Detail"))
    {
        util::log("RecurrentLayer::Detail") << "  weight grad: "
            << weightGradients.debugString();
    }

    gradients.push_back(std::move(weightGradients));

    inputDeltaVector.push_back(reshape(inputDeltas, {getInputCount(), miniBatch, timesteps}));
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
    return _weights.precision();
}

double RecurrentLayer::computeWeightCost() const
{
    return getWeightCostFunction()->getCost(_weights);
}

Dimension RecurrentLayer::getInputSize() const
{
    return {_layerSize, 1, 1};
}

Dimension RecurrentLayer::getOutputSize() const
{
    if(_direction == matrix::RECURRENT_BIDIRECTIONAL)
    {
        return {2 * _layerSize, 1, 1};
    }
    else
    {
        return getInputSize();
    }
}

size_t RecurrentLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t RecurrentLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t RecurrentLayer::totalNeurons() const
{
    return getInputCount() * _layers;
}

size_t RecurrentLayer::totalConnections() const
{
    return _weights.elements();
}

size_t RecurrentLayer::getFloatingPointOperationCount() const
{
    return 2 * totalConnections();
}

size_t RecurrentLayer::getActivationMemory() const
{
    return getOutputCount() * precision().size();
}

void RecurrentLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["weights"] = properties.path() + "." + properties.key() + ".weights.npy";

    properties["layer-size"] = std::to_string(_layerSize);
    properties["layers"]     = std::to_string(_layers);

    properties["direction"]  = std::to_string(_direction);
    properties["layer-type"] = std::to_string(_layerType);
    properties["input-mode"] = std::to_string(_inputMode);

    saveToArchive(archive, properties["weights"], _weights);

    saveLayer(archive, properties);
}

void RecurrentLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _weights = matrix::loadFromArchive(archive, properties["weights"]);

    _layerSize = properties.get<size_t>("layer-size");
    _layers    = properties.get<size_t>("layers");

    _direction = properties.get<int>("direction");
    _layerType = properties.get<int>("layer-type");
    _inputMode = properties.get<int>("input-mode");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> RecurrentLayer::clone() const
{
    return std::make_unique<RecurrentLayer>(*this);
}

std::string RecurrentLayer::getTypeName() const
{
    return "RecurrentLayer";
}

}

}



