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
: _beamSize(beamSize),
  _costFunctionName(costFunctionName),
  _costFunctionWeight(costFunctionWeight),
  _inputSize(std::make_unique<matrix::Dimension>(inputSize)),
  _precision(std::make_unique<matrix::Dimension>(precision))
{

}

CTCDecoderLayer::~CTCDecoderLayer()
{
    // intentionally blank
}

CTCDecoderLayer::CTCDecoderLayer(const CTCDecoderLayer& l)
: Layer(l),
  _beamSize(l._beamSize),
  _costFunctionName(l._costFunctionName),
  _costFunctionWeight(l._costFunctionWeight),
  _inputSize(std::make_unique<matrix::Dimension>(*l._inputSize)),
  _precision(std::make_unique<matrix::Precision>(*l._precision))
{


}

CTCDecoderLayer& CTCDecoderLayer::operator=(const CTCDecoderLayer& layer)
{
    Layer::operator=(l);

    if(this == &layer)
    {
        return *this;
    }

    _beamSize           = l._beamSize;
    _costFunctionName   = l._costFunctionName;
    _costFunctionWeight = l._costFunctionWeight;
    _inputSize          = std::make_unique<matrix::Dimension>(*l._inputSize);
    _precision          = std::make_unique<matrix::Precision>(*l._precision);

    return *this;
}

void CTCDecoderLayer::initialize()
{
    // intentionally blank
}

void CTCDecoderLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    auto beamSearchOutputSize = matrix::getBeamSearchOutputSize(
        inputActivations.size(), _beamSize);

    Matrix outputActivations(beamSearchOutputSize, inputActivations.precision());
    Matrix outputActivationWeights({_beamSize, miniBatchSize}, inputActivations.precision());

    matrix::beamSearch(outputActivationWeights, outputActivations, inputActivations, _beamSize);

    saveMatrix("outputActivationWeights", outputActivationWeights);

    if(util::isLogEnabled("CTCDecoderLayer::Detail"))
    {
        util::log("CTCDecoderLayer::Detail") << "  output activation: "
            << outputActivations.debugString();
        util::log("CTCDecoderLayer::Detail") << "  output activation weights: "
            << outputActivationWeights.debugString();
    }
    else
    {
        util::log("CTCDecoderLayer") << "  output activation size: "
            << outputActivations.shapeString() << "\n";
    }

    outputActivationsVector.push_back(outputActivations);
}

void CTCDecoderLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients         = bundle[   "gradients"].get<matrix::MatrixVector>();
    auto& inputDeltaVector  = bundle[ "inputDeltas"].get<matrix::MatrixVector>();
    auto& outputDeltaVector = bundle["outputDeltas"].get<matrix::MatrixVector>();

    assert(outputDeltaVector.size() == 1);

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

