/*  \file   CTCDecoderLayer.cpp
    \author Gregory Diamos
    \date   October 9, 2016
    \brief  The implementation of the CTCDecoderLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/CTCDecoderLayer.h>
#include <lucius/network/interface/Bundle.h>
#include <lucius/network/interface/CostFunctionFactory.h>
#include <lucius/network/interface/CostFunction.h>

#include <lucius/matrix/interface/CTCOperations.h>

#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/Precision.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/PropertyTree.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace network
{

typedef matrix::Dimension Dimension;
typedef matrix::Precision Precision;
typedef matrix::LabelVector LabelVector;
typedef matrix::IndexVector IndexVector;
typedef matrix::MatrixVector MatrixVector;
typedef matrix::Matrix Matrix;

class CTCDecoderLayerImplementation
{
public:
    CTCDecoderLayerImplementation(const Dimension& inputSize, size_t beamSize,
        const std::string& costFunctionName, double costFunctionWeight,
        const matrix::Precision& precision)
    : _beamSize(beamSize),
      _costFunctionName(costFunctionName),
      _costFunctionWeight(costFunctionWeight),
      _inputSize(inputSize),
      _precision(precision)
    {

    }

public:
    void setLabels(const LabelVector& labels)
    {
        _labels = labels;
    }

    void setTimesteps(const IndexVector& timesteps)
    {
        _timesteps = timesteps;
    }

public:
    size_t getBeamSize() const
    {
        return _beamSize;
    }

    const Dimension& getInputSize() const
    {
        return _inputSize;
    }

    const Precision& getPrecision() const
    {
        return _precision;
    }

public:
    void setBeamSize(size_t beamSize)
    {
        _beamSize = beamSize;
    }

    void setInputSize(const Dimension& d)
    {
        _inputSize = d;
    }

    void setPrecision(const Precision& p)
    {
        _precision = p;
    }

public:
    const std::string& getCostFunctionName() const
    {
        return _costFunctionName;
    }

    double getCostFunctionWeight() const
    {
        return _costFunctionWeight;
    }

public:
    void clearReversePropagationData()
    {
        _labels.clear();
        _timesteps.clear();
    }

public:
    void saveInputLabels(const LabelVector& labels)
    {
        _labels = labels;
    }

    void saveInputTimesteps(const IndexVector& timesteps)
    {
        _timesteps = timesteps;
    }

public:
    const LabelVector& getInputLabels() const
    {
        return _labels;
    }

    const IndexVector& getInputTimesteps() const
    {
        return _timesteps;
    }

private:
    LabelVector _labels;
    IndexVector _timesteps;

private:
    size_t _beamSize;

private:
    std::string _costFunctionName;
    double      _costFunctionWeight;

private:
    Dimension _inputSize;
    Precision _precision;
};

CTCDecoderLayer::CTCDecoderLayer()
: CTCDecoderLayer({1, 1, 1})
{

}

CTCDecoderLayer::CTCDecoderLayer(const Dimension& inputSize)
: CTCDecoderLayer(inputSize, util::KnobDatabase::getKnobValue("CTCDecoderLayer::BeamSize", 128))
{

}

CTCDecoderLayer::CTCDecoderLayer(const Dimension& inputSize, size_t beamSize)
: CTCDecoderLayer(inputSize, beamSize, "CTCCostFunction", 1.0,
  matrix::Precision::getDefaultPrecision())
{

}

CTCDecoderLayer::CTCDecoderLayer(const Dimension& inputSize, size_t beamSize,
    const std::string& costFunctionName, double costFunctionWeight,
    const matrix::Precision& precision)
: _implementation(std::make_unique<CTCDecoderLayerImplementation>(
    inputSize, beamSize, costFunctionName, costFunctionWeight, precision))
{

}

CTCDecoderLayer::~CTCDecoderLayer()
{
    // intentionally blank
}

CTCDecoderLayer::CTCDecoderLayer(const CTCDecoderLayer& l)
: Layer(l),
  _implementation(std::make_unique<CTCDecoderLayerImplementation>(*l._implementation))
{


}

CTCDecoderLayer& CTCDecoderLayer::operator=(const CTCDecoderLayer& layer)
{
    Layer::operator=(layer);

    if(this == &layer)
    {
        return *this;
    }

    *_implementation = *layer._implementation;

    return *this;
}

void CTCDecoderLayer::initialize()
{
    // intentionally blank
}

// Note that the convention here is for the 0th char to be the start/stop
static void expandLabels(Bundle& bundle, size_t beamSize)
{
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    auto& outputActivations = outputActivationsVector.back();

    auto& labels = bundle["referenceLabels"].get<LabelVector>();

    size_t characters    = outputActivations.size().front();
    size_t miniBatchSize = outputActivations.size()[outputActivations.size().size() - 2];
    size_t maxTimesteps  = outputActivations.size()[outputActivations.size().size() - 1];

    auto referenceActivations = matrix::zeros({characters, miniBatchSize, maxTimesteps},
        outputActivations.precision());

    for(size_t miniBatch = 0; miniBatch < miniBatchSize; ++miniBatch)
    {
        referenceActivations(0, miniBatch, 0) = 1.0;

        for(size_t timestep = 0; timestep < maxTimesteps; ++timestep)
        {
            if(timestep < labels[miniBatch/beamSize].size())
            {
                referenceActivations(timestep + 1, miniBatch,
                    labels[miniBatch/beamSize][timestep]) = 1.0;
            }
            else
            {
                referenceActivations(timestep + 1, miniBatch, 0) = 1.0;
            }
        }
    }

    bundle["referenceActivations"] = MatrixVector({referenceActivations});
}

static void reshapeOutputActivations(Matrix& outputActivations, const Dimension& inputSize,
    size_t beamSize)
{
    Dimension outputSize = inputSize;

    outputSize[inputSize.size() - 2] *= beamSize;

    outputActivations = reshape(outputActivations, outputSize);
}

void CTCDecoderLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    saveMatrix("inputActivations", inputActivations);

    auto beamSearchOutputSize = matrix::getBeamSearchOutputSize(
        inputActivations.size(), _implementation->getBeamSize());

    size_t miniBatchSize = inputActivations.size()[inputActivations.size().size() - 2];
    size_t timesteps     = inputActivations.size()[inputActivations.size().size() - 1];

    Matrix inputPaths({_implementation->getBeamSize(), miniBatchSize, timesteps},
        inputActivations.precision());
    Matrix outputActivations(beamSearchOutputSize, inputActivations.precision());
    Matrix outputActivationWeights({_implementation->getBeamSize(), miniBatchSize},
        inputActivations.precision());

    matrix::ctcBeamSearch(outputActivationWeights, inputPaths, outputActivations,
        inputActivations, _implementation->getBeamSize());

    reshapeOutputActivations(outputActivations, inputActivations.size(),
        _implementation->getBeamSize());

    saveMatrix("outputActivationWeights", outputActivationWeights);
    saveMatrix("inputPaths",              inputPaths);

    if(util::isLogEnabled("CTCDecoderLayer::Detail"))
    {
        util::log("CTCDecoderLayer::Detail") << "  output activation: "
            << outputActivations.debugString();
        util::log("CTCDecoderLayer::Detail") << "  input paths: "
            << inputPaths.debugString();
        util::log("CTCDecoderLayer::Detail") << "  output activation weights: "
            << outputActivationWeights.debugString();
    }
    else
    {
        util::log("CTCDecoderLayer") << "  output activation size: "
            << outputActivations.shapeString() << "\n";
        util::log("CTCDecoderLayer") << "  input paths: "
            << inputPaths.shapeString() << "\n";
        util::log("CTCDecoderLayer") << "  output activation weights: "
            << outputActivationWeights.shapeString() << "\n";
    }

    _implementation->saveInputLabels(bundle["referenceLabels"].get<LabelVector>());
    _implementation->saveInputTimesteps(bundle["inputTimesteps"].get<IndexVector>());

    outputActivationsVector.push_back(outputActivations);

    expandLabels(bundle, _implementation->getBeamSize());
}

Matrix computeCtcInputDeltas(const std::string& costFunctionName,
    double costFunctionWeight, const Matrix& inputActivations, const LabelVector& inputLabels,
    const IndexVector& inputTimesteps)
{
    auto costFunction = CostFunctionFactory::create(costFunctionName);

    Bundle bundle;

    bundle["outputActivations"] = MatrixVector({inputActivations});
    bundle["referenceLabels"]   = inputLabels;
    bundle["inputTimesteps"]    = inputTimesteps;

    costFunction->computeDelta(bundle);

    auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

    auto& outputDeltas = outputDeltasVector.front();

    return apply(outputDeltas, matrix::Multiply(costFunctionWeight));
}

void CTCDecoderLayer::runReverseImplementation(Bundle& bundle)
{
    auto& inputDeltaVector  = bundle[ "inputDeltas"].get<matrix::MatrixVector>();
    auto& outputDeltaVector = bundle["outputDeltas"].get<matrix::MatrixVector>();

    assert(outputDeltaVector.size() == 1);

    auto outputDeltas = outputDeltaVector.front();
    auto outputActivationWeights = loadMatrix("outputActivationWeights");
    auto inputPaths = loadMatrix("inputPaths");

    if(util::isLogEnabled("CTCDecoderLayer::Detail"))
    {
        util::log("CTCDecoderLayer::Detail") << "  output deltas: "
            << outputDeltas.debugString();
    }
    else
    {
        util::log("CTCDecoderLayer") << "  output deltas size: "
            << outputDeltas.shapeString() << "\n";
    }

    auto inputActivations = loadMatrix("inputActivations");

    Matrix beamSearchInputDeltas(inputActivations.size(), inputActivations.precision());

    matrix::ctcBeamSearchInputGradients(beamSearchInputDeltas, outputActivationWeights,
        inputPaths, outputDeltas, _implementation->getBeamSize());

    if(util::isLogEnabled("CTCDecoderLayer::Detail"))
    {
        util::log("CTCDecoderLayer::Detail") << "  beam search output deltas: "
            << outputDeltas.debugString();
    }
    else
    {
        util::log("CTCDecoderLayer") << "  beam search output deltas size: "
            << outputDeltas.shapeString() << "\n";
    }

    auto inputLabels = _implementation->getInputLabels();
    auto inputTimesteps = _implementation->getInputTimesteps();

    auto ctcInputDeltas = computeCtcInputDeltas(_implementation->getCostFunctionName(),
        _implementation->getCostFunctionWeight(),
        inputActivations, inputLabels, inputTimesteps);

    auto inputDeltas = apply(Matrix(beamSearchInputDeltas), ctcInputDeltas, matrix::Add());

    inputDeltaVector.push_back(inputDeltas);
}

MatrixVector& CTCDecoderLayer::weights()
{
    static MatrixVector w;
    return w;
}

const MatrixVector& CTCDecoderLayer::weights() const
{
    static MatrixVector w;
    return w;
}

const matrix::Precision& CTCDecoderLayer::precision() const
{
    return _implementation->getPrecision();
}

double CTCDecoderLayer::computeWeightCost() const
{
    return 0.0;
}

Dimension CTCDecoderLayer::getInputSize() const
{
    return _implementation->getInputSize();
}

Dimension CTCDecoderLayer::getOutputSize() const
{
    auto outputSize = getInputSize();

    assert(outputSize.size() >= 3);

    size_t& minibatch = outputSize[outputSize.size() - 2];

    minibatch *= _implementation->getBeamSize();

    return outputSize;
}

size_t CTCDecoderLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t CTCDecoderLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t CTCDecoderLayer::totalNeurons() const
{
    return 0;
}

size_t CTCDecoderLayer::totalConnections() const
{
    return totalNeurons();
}

size_t CTCDecoderLayer::getFloatingPointOperationCount() const
{
    return getInputCount() * getInputCount();
}

size_t CTCDecoderLayer::getActivationMemory() const
{
    return precision().size() * getOutputCount();
}

void CTCDecoderLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    properties["beam-size"]  = std::to_string(_implementation->getBeamSize());
    properties["input-size"] = _implementation->getInputSize().toString();
    properties["precision"]  = _implementation->getPrecision().toString();

    saveLayer(archive, properties);
}

void CTCDecoderLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    _implementation->setInputSize(Dimension::fromString(properties["input-size"]));
    _implementation->setPrecision(*matrix::Precision::fromString(properties["precision"]));
    _implementation->setBeamSize(properties.get<size_t>("beam-size"));

    loadLayer(archive, properties);
}

void CTCDecoderLayer::clearReversePropagationData()
{
    _implementation->clearReversePropagationData();

    Layer::clearReversePropagationData();
}

std::unique_ptr<Layer> CTCDecoderLayer::clone() const
{
    return std::unique_ptr<Layer>(new CTCDecoderLayer(*this));
}

std::string CTCDecoderLayer::getTypeName() const
{
    return "CTCDecoderLayer";
}

}
}

