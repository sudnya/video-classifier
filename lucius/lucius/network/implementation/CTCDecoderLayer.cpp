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
#include <lucius/util/interface/string.h>
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

    bool hasCostFunction() const
    {
        return getCostFunctionWeight() > 0.0;
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

static void expandLabels(Bundle& bundle, size_t beamSize)
{
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    auto& outputActivations = outputActivationsVector.back();

    auto& labels = bundle["referenceLabels"].get<LabelVector>();
    auto& inputTimesteps = bundle["inputTimesteps"].get<IndexVector>();

    size_t characters    = outputActivations.size().front();
    size_t miniBatchSize = outputActivations.size()[outputActivations.size().size() - 2];
    size_t maxTimesteps  = outputActivations.size()[outputActivations.size().size() - 1];

    size_t originalMiniBatchSize = miniBatchSize / beamSize;

    LabelVector expandedLabels;
    IndexVector expandedTimesteps;

    for(size_t miniBatch = 0; miniBatch < originalMiniBatchSize; ++miniBatch)
    {
        for(size_t beam = 0; beam < beamSize; ++beam)
        {
            expandedLabels.push_back(labels[miniBatch]);
            expandedTimesteps.push_back(inputTimesteps[miniBatch]);
        }
    }

    labels = std::move(expandedLabels);
    inputTimesteps = std::move(expandedTimesteps);

    auto referenceActivations = matrix::zeros({characters, miniBatchSize, maxTimesteps},
        outputActivations.precision());

    for(size_t miniBatch = 0; miniBatch < miniBatchSize; ++miniBatch)
    {
        for(size_t timestep = 0; timestep < maxTimesteps; ++timestep)
        {
            if(timestep < labels[miniBatch].size())
            {
                size_t character = labels[miniBatch][timestep];

                assert(character < characters);

                referenceActivations(character, miniBatch, timestep) = 1.0;
            }
            else
            {
                referenceActivations(characters - 1, miniBatch, timestep) = 1.0;
            }
        }
    }

    if(util::isLogEnabled("CTCDecoderLayer::Detail"))
    {
        util::log("CTCDecoderLayer::Detail") << "  reference labels: \n";
        for(auto& label : labels)
        {
            util::log("CTCDecoderLayer::Detail") << "   reference label: "
                << util::toString(label) << "\n";

        }
        util::log("CTCDecoderLayer::Detail") << "  reference activations: "
            << referenceActivations.debugString();
    }
    else
    {
        util::log("CTCDecoderLayer::Detail") << "  reference labels size: "
            << labels.size() << "\n";
        util::log("CTCDecoderLayer") << "  reference activations size: "
            << outputActivations.shapeString() << "\n";
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

static void computeCtcCosts(Bundle& bundle, const std::string& costFunctionName,
    double costFunctionWeight,
    const Matrix& inputActivations, const LabelVector& labels, const IndexVector& inputTimesteps)
{
    auto costFunction = CostFunctionFactory::create(costFunctionName);

    Bundle inputBundle;

    inputBundle["outputActivations"] = MatrixVector({inputActivations});
    inputBundle["referenceLabels"]   = labels;
    inputBundle["inputTimesteps"]    = inputTimesteps;

    costFunction->computeCost(inputBundle);

    auto costs = inputBundle["costs"].get<Matrix>();

    bundle["cost"] = static_cast<double>(reduce(costs, {}, matrix::Add())[0]) * costFunctionWeight;
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

    bundle["outputActivationWeights"] = outputActivationWeights;

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

    _implementation->saveInputLabels(   bundle["referenceLabels"].get<LabelVector>());
    _implementation->saveInputTimesteps(bundle["inputTimesteps" ].get<IndexVector>());

    outputActivationsVector.push_back(outputActivations);

    expandLabels(bundle, _implementation->getBeamSize());

    if(_implementation->hasCostFunction())
    {
        auto inputTimesteps = _implementation->getInputTimesteps();
        auto inputLabels    = _implementation->getInputLabels();

        computeCtcCosts(bundle, _implementation->getCostFunctionName(),
            _implementation->getCostFunctionWeight(),
            inputActivations, inputLabels, inputTimesteps);
    }
}

static Matrix computeCtcInputDeltas(const std::string& costFunctionName,
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
    auto& inputDeltaVector = bundle["inputDeltas"].get<matrix::MatrixVector>();
    auto& outputActivationWeightDeltas =
        bundle["outputActivationWeightDeltas"].get<matrix::Matrix>();

    auto outputActivationWeights = loadMatrix("outputActivationWeights");
    auto inputPaths = loadMatrix("inputPaths");

    auto inputActivations = loadMatrix("inputActivations");

    Matrix inputDeltas = zeros(inputActivations.size(), inputActivations.precision());

    matrix::ctcBeamSearchInputGradients(inputDeltas, outputActivationWeights,
        inputPaths, outputActivationWeightDeltas, inputActivations);

    if(util::isLogEnabled("CTCDecoderLayer::Detail"))
    {
        util::log("CTCDecoderLayer::Detail") << "  beam search input deltas: "
            << inputDeltas.debugString();
    }
    else
    {
        util::log("CTCDecoderLayer") << "  beam search input deltas size: "
            << inputDeltas.shapeString() << "\n";
    }

    if(_implementation->hasCostFunction())
    {
        auto inputTimesteps = _implementation->getInputTimesteps();
        auto inputLabels    = _implementation->getInputLabels();

        auto ctcInputDeltas = computeCtcInputDeltas(_implementation->getCostFunctionName(),
            _implementation->getCostFunctionWeight(),
            inputActivations, inputLabels, inputTimesteps);

        // TODO: see if it is possible to remove this line, it is only necessary because
        //       by CTC can't tell the difference between a beam and a minibatch
        size_t miniBatchSize = inputActivations.size()[1];

        apply(ctcInputDeltas, ctcInputDeltas, matrix::Multiply(miniBatchSize));

        if(util::isLogEnabled("CTCDecoderLayer::Detail"))
        {
            util::log("CTCDecoderLayer::Detail") << "  ctc input deltas: "
                << ctcInputDeltas.debugString();
        }
        else
        {
            util::log("CTCDecoderLayer") << "  ctc input deltas size: "
                << ctcInputDeltas.shapeString() << "\n";
        }

        apply(inputDeltas, inputDeltas, ctcInputDeltas, matrix::Add());

        if(util::isLogEnabled("CTCDecoderLayer::Detail"))
        {
            util::log("CTCDecoderLayer::Detail") << "  combined input deltas: "
                << inputDeltas.debugString();
        }
        else
        {
            util::log("CTCDecoderLayer") << "  combined input deltas size: "
                << inputDeltas.shapeString() << "\n";
        }
    }

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
    auto inputSize = _implementation->getInputSize();

    inputSize[inputSize.size() - 1] = 1;
    inputSize[inputSize.size() - 2] = 1;

    return inputSize;
}

Dimension CTCDecoderLayer::getOutputSize() const
{
    return getInputSize();
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

