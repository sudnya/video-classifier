/*  \file   CTCDecoderLayer.cpp
    \author Gregory Diamos
    \date   October 9, 2016
    \brief  The implementation of the CTCDecoderLayer class.
*/

// Lucius Includes
#include <lucius/interface/network/CTCDecoderLayer.h>

namespace lucius
{
namespace network
{

class CTCDecoderLayerImplementation
{
public:
    CTCDecoderLayerImplementation(const Dimension& inputSize, size_t beamSize,
        const std::string& costFunctionName, double costFunctionWeight,
        const matrix::Precision& precision)
    : _beamSize(beamSize),
      _costFunctionName(costFunctioName),
      _costFunctionWeight(costFunctionWeight),
      _inputSize(inputSize),
      _precision(precision)
    {

    }

public:
    void setLabels(const LabelVector& labels);
    void setTimesteps(const IndexVector& timesteps);

private:
    LabelVector _labels;
    IndexVector _timesteps;

private:
    size_t _beamSize;

private:
    std::string _costFunctionName;
    double      _costFunctionWeight;

private:
    matrix::Dimension _inputSize;
    matrix::Precision _precision;
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
: CTCDecoderLayer(inputSize, beamSize, matrix::Precision::getDefaultPrecision())
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
: Layer(l), _implementation(std::make_unique<CTCDeoderLayerImplementation>(*l._implementation))
{


}

CTCDecoderLayer& CTCDecoderLayer::operator=(const CTCDecoderLayer& layer)
{
    Layer::operator=(l);

    if(this == &layer)
    {
        return *this;
    }

    *_implementation = *l._implementation;

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

    size_t miniBatchSize = outputActivations.size()[outputActivations.size().size() - 2];
    size_t maxTimesteps  = outputActivations.size()[outputActivations.size().size() - 1];

    auto referenceActivations = matrix::zeros({characters, miniBatchSize, maxTimesteps},
        inputActivations.precision());

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

    bundle["referenceActivations"] = referenceActivations;
}

void CTCDecoderLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    auto beamSearchOutputSize = matrix::getBeamSearchOutputSize(
        inputActivations.size(), _beamSize);

    Matrix inputPaths(beamSearchOutputSize, inputActivations.precision());
    Matrix outputActivations(beamSearchOutputSize, inputActivations.precision());
    Matrix outputActivationWeights({_beamSize, miniBatchSize}, inputActivations.precision());

    matrix::ctcBeamSearch(outputActivationWeights, inputPaths, outputActivations,
        inputActivations, _beamSize);

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

    outputActivationsVector.push_back(outputActivations);

    expandLabels(bundle, _beamSize);
}

void CTCDecoderLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients         = bundle[   "gradients"].get<matrix::MatrixVector>();
    auto& inputDeltaVector  = bundle[ "inputDeltas"].get<matrix::MatrixVector>();
    auto& outputDeltaVector = bundle["outputDeltas"].get<matrix::MatrixVector>();

    assert(outputDeltaVector.size() == 1);

    auto outputDeltas = outputDeltasVector.front();
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

    auto beamSearchInputDeltas = matrix::ctcBeamSearchInputGradients(outputActivationWeights,
        inputPaths, outputDeltas, _beamSize);

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

    auto inputActivations = loadMatrix("inputActivations");
    auto inputLabels = loadInputLabels();
    auto inputTimesteps = loadInputTimesteps();

    auto ctcInputDeltas = computeCtcInputDeltas(_costFunctionName, _costFunctionWeight,
        inputActivations, inputLabels, inputTimesteps);

    auto inputDeltas = apply(beamSearchInputDeltas, ctcInputDeltas, matrix::Add());

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
    return *_precision;
}

double CTCDecoderLayer::computeWeightCost() const
{
    return 0.0;
}

Dimension CTCDecoderLayer::getInputSize() const
{
    return *_inputSize;
}

Dimension CTCDecoderLayer::getOutputSize() const
{
    auto outputSize = getInputSize();

    size_t timesteps = outputSize[outputSize.size() - 1];
    size_t minibatch = outputSize[outputSize.size() - 2];

    outputSize.push_back(timesteps);
    outputSize[outputSize[outputSize.size() - 3]] = 1;
    outputSize[outputSize[outputSize.size() - 2]] = minibatch;

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
    properties["beam-size"]  = std::to_string(_beamSize);
    properties["input-size"] = _inputSize->toString();
    properties["precision"]  = _precision->toString();

    saveLayer(archive, properties);
}

void CTCDecoderLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    *_inputSize  = Dimension::fromString(properties["input-size"]);
    _precision   = matrix::Precision::fromString(properties["precision"]);
    _beamSize    = properties.get<size_t>("beam-size");

    loadLayer(archive, properties);
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

