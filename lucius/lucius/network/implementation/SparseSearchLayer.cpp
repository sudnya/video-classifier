/*  \file   SparseSearchLayer.cpp
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SparseSearchLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/SparseSearchLayer.h>

namespace lucius
{
namespace network
{

SparseSearchLayer::SparseSearchLayer()
: SparseSearchLayer(0, 0, 0, 0)
{

}

SparseSearchLayer::SparseSearchLayer(size_t depth, size_t size, size_t radix, size_t selections)
: SparseSearchLayer(depth, size, radix, selections, matrix::Precision::getDefaultPrecision())
{

}

static std::unique_ptr<SubgraphLayer> createSelectUnit(size_t depth, size_t size,
    size_t radix, size_t selections)
{
    auto layer = std::make_unique<SubgraphLayer>();

    for(size_t i = 0; i < depth; ++i)
    {
        size_t inputSize  = size;
        size_t outputSize = size;

        if(i + 1 == depth)
        {
            outputSize = size + radix;
        }

        std::stringstream layerName;

        layerName << "select" << i;

        layer.addLayer(layerName.str(), LayerFactor::create("FeedForwardLayer",
            {{"InputSize", inputSize}, {"OutputSize", outputSize}}));
    }

    return layer;
}

static std::unique_ptr<SubgraphLayer> createProcessUnit(size_t depth, size_t size, size_t radix,
    size_t selections)
{
    auto layer = std::make_unique<SubgraphLayer>();

    size_t processedSize       = computeProcessedSize(size,        selections);
    size_t generatedSize       = computeGeneratedSize(size, radix, selections);
    size_t processedTimeSize   = computeProcessedTimeSize(size);
    size_t selectToProcessSize = computeSelectToProcessSize(size);

    for(size_t i = 0; i < depth; ++i)
    {
        size_t inputSize  = size;
        size_t outputSize = size;

        if(i == 0)
        {
            inputSize = generatedSize + processedTimeSize + selectToProcessSize + generatedSize;
        }

        if(i + 1 == depth)
        {
            outputSize = processedSize + processedTimeSize;
        }

        std::stringstream layerName;

        layerName << "process" << i;

        layer.addLayer(layerName.str(), LayerFactor::create("FeedForwardLayer",
            {{"InputSize", inputSize}, {"OutputSize", outputSize}}));
    }

    return layer;
}

SparseSearchLayer::SparseSearchLayer(size_t depth, size_t size, size_t radix, size_t selections,
    const matrix::Precision& precision)
: _selectUnit(createSelectUnit(depth, size, radix, selections)),
  _processUnit(createProcessUnit(depth, size, radix, selections)),
  _depth(depth),
  _size(size),
  _radix(radix),
  _selections(selections)
{

}

SparseSearchLayer::~SparseSearchLayer()
{

}

SparseSearchLayer::SparseSearchLayer(const SparseSearchLayer& l)
: Layer(l),
  _selectUnit(l._selectUnit->clone()),
  _processUnit(l._processUnit->clone()),
  _depth(l._depth),
  _size(l._size),
  _radix(l._radix),
  _selections(l._selections)
{

}

SparseSearchLayer& SparseSearchLayer::operator=(const SparseSearchLayer& l)
{
    Layer::operator=(l);

    if(this == &l)
    {
        return *this;
    }

    _selectUnit  = std::move(l._selectUnit->clone());
    _processUnit = std::move(l._processUnit->clone());

    _depth      = l._depth;
    _size       = l._size;
    _radix      = l._radix;
    _selections = l._selections;

    return *this;
}

void SparseSearchLayer::initialize()
{
    _selectUnit->initialize();
    _processUnit->initialize();
}

static MatrixVector runSelectForward(SubgraphLayer& layer,
    const MatrixVector& inputActivations, size_t size)
{
    auto extendedSize = inputActivation.size();
    extendedSize[0]   = computeSelectInputSize(size);

    Matrix extendedInput(extendedSize, inputActivatiobs.back().precision());

    auto begin  = zeros(extendedSize);
    auto middle = Dimension(computeSelectCommandSize(size), extendedSize[1], extendedSize[2]);
    auto end    = extendedSize;

    auto previousTimestepSlice = slice(extendedInput, begin,  middle);
    auto currentTimestepSlice  = slice(extendedInput, middle, end);

    if(inputActivations.size() == 2)
    {
        auto previousTimestepActivation = inputActivations[0];
        auto inputActivation            = inputActivations[1];

        copy(previousTimestepSlice, previousTimestepActivation);
        copy(currentTimestepSlice,  inputActivation);
    }
    else
    {
        auto inputActivation = inputActivations[0];
        auto previousTimestepActivation = zeros(inputActivation.size());

        copy(previousTimestepSlice, previousTimestepActivation);
        copy(currentTimestepSlice,  inputActivation);
    }

    MatrixVector result;

    layer.runForwardImplementation(result, inputActivation);

    return result;
}

static MatrixVector runRecursionForward(SparseSearchLayer& layer, InputOutputEngine& engine,
    const MatrixVector& selectResult)
{
    // select based on the select unit's output
    auto selectedSegments = computeSelectedSegments(layer, selectResult);

    if(isLastRecursionLevel(layer, engine))
    {
        // read data directly from the engine
        Matrix resultData({radix * unitSize, 1, 1}, selectResult.front().precision());

        size_t index = 0;
        for(auto& selectedSegment : selectedSegments)
        {
            auto resultSlice = slice(resultData, {index * unitSize, 0, 0},
                {(index + 1) * unitSize, 1, 1});

            auto loadedData = engine.getSlice({selectedSegment}, {selectedSegment + unitSize});

            copy(resultSlice, loadedData);

            ++index;
        }

        MatrixVector result;

        result.push_back(resultData);

        return result;
    }
    else
    {
        size_t originalSegmentSize = layer.getSegmentSize()
        size_t newSegmentSize      = reduceSegmentSize(originalSegmentSize);

        layer.setSegmentSize(newSegmentSize);

        MatrixVector result;

        // recurse on the selected elements
        for(auto& selectedSegment : selectedSegments)
        {
            layer.setSegment(selectedSegment);

            layer.runForwardImplementation(resultSlice, selectResult);
        }

        layer.setSegmentSize(originalSegmentSize);

        return result;
    }
}

static MatrixVector runRouterForward(const MatrixVector& selectResult,
    const MatrixVector& dataForSelectedUnits)
{
    auto data         = dataForSelectedUnits.front();
    auto selectResult = sliceOutSelections(selectResult.front());

    auto reshapedData = reshape(data, {unitSize, units, 1, 1});

    MatrixVector result;

    result.push_back(broadcast(matrix::Multiply(), selectResult, reshapedData));

    return result;
}

static MatrixVector runProcessForward(SungraphLayer& layer, const MatrixVector& selectResult,
    const MatrixVector& generatedData, const MatrixVector& inputActivations)
{
    auto extendedSize = inputActivation.size();
    extendedSize[0]   = computeProcessInputSize(size);

    Matrix extendedInput(extendedSize, inputActivatiobs.back().precision());

    auto begin  = zeros(extendedSize);
    auto middle = Dimension(computeSelectCommandSize(size), extendedSize[1], extendedSize[2]);
    auto end    = extendedSize;

    auto previousTimestepSlice = slice(extendedInput, begin,  middle);
    auto currentTimestepSlice  = slice(extendedInput, middle, end);

    if(inputActivations.size() == 2)
    {
        auto previousTimestepActivation = inputActivations[0];
        auto inputActivation            = inputActivations[1];

        copy(previousTimestepSlice, previousTimestepActivation);
        copy(currentTimestepSlice,  inputActivation);
    }
    else
    {
        auto inputActivation = inputActivations[0];
        auto previousTimestepActivation = zeros(inputActivation.size());

        copy(previousTimestepSlice, previousTimestepActivation);
        copy(currentTimestepSlice,  inputActivation);
    }

    MatrixVector result;

    layer.runForward(result, inputActivation);

    return result;
}

void SparseSearchLayer::runForwardImplementation(MatrixVector& outputActivations,
    const MatrixVector& inputActivations)
{
    // Evaluate the select unit
    auto selectResult = runSelectForward(*_selectUnit, inputActivations, _size);

    // Recurse based on the selected units
    auto dataForSelectedUnits = runRecursionForward(*this, *_engine, selectResult);

    // Evaluate the router unit
    auto generatedData = runRouterForward(selectResult, dataForSelectedUnits);

    // Evaluate the process unit
    auto processedData = runProcessForward(*_processUnit, selectResult, generatedData);

    outputActivations.push_back(std::move(processedData));
}

static MatrixVector runProcessReverse(SubgrahLayer& layer,
    MatrixVector& gradients, const MatrixVector& outputDeltas)
{
    MatrixVector generatedDataDeltas;
    MatrixVector packedOutputDeltas;

    packOutputDeltas(packedOutputDeltas, outputDeltas);

    layer.runReverse(gradients, generatedDataDeltas, packedOutputDelta);

    return generatedDataDeltas;
}

static MatrixVector runRecursionReverse(SparseSearchLayer& layer, MatrixVector& gradients,
    const MatrixVector& generatedDataDeltas)
{
    assertM(false, "Not Implemented");

    MatrixVector result;

    return result;
}

static MatrixVector runSelectReverse(MatrixVector& gradients, const MatrixVector& generatedDataDeltas,
    const MatrixVector& selectedAndGeneratedDeltas)
{
    assertM(false, "Not Implemented");

    MatrixVector result;

    return result;
}

Matrix SparseSearchLayer::runReverseImplementation(MatrixVector& gradients,
    MatrixVector& inputDeltas,
    const Matrix& outputDeltas)
{
    // Evaluate the process unit
    auto generatedDataDeltas = runProcessReverse(*_processUnit, gradients, outputDeltas);

    // Evaluate the router unit
    auto selectedAndGeneratedDelats = runRouterReverse(gradients, generatedDataDeltas);

    // Unwind the recursion
    auto selectDeltas = runRecursionReverse(this, gradients, generatedDataDeltas);

    // Evaluate the select unit
    auto selectInputDeltas = runSelectReverse(gradients, generatedDataDeltas,
        selectedAndGeneratedDeltas);

    inputDeltas.push_back(selectInputDeltas);
}

MatrixVector& SparseSearchLayer::weights()
{
    return _parameters;
}

const MatrixVector& SparseSearchLayer::weights() const
{
    return _parameters;
}

const matrix::Precision& SparseSearchLayer::precision() const
{
    return _selectUnit->precision();
}

double SparseSearchLayer::computeWeightCost() const
{
    return _selectUnit->computeWeightCost() + _processUnit->computeWeightCost();
}

Dimension SparseSearchLayer::getInputSize() const
{
    return {computeCommandSize(_size), 1, 1};
}

Dimension SparseSearchLayer::getOutputSize() const
{
    return {computeProcessedSize(_size, _selected), 1, 1};
}

size_t SparseSearchLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t SparseSearchLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t SparseSearchLayer::totalNeurons() const
{
    return _selectUnit->totalNeurons() + _processUnit->totalNeurons();
}

size_t SparseSearchLayer::totalConnections() const
{
    return _selectUnit->totalConnections() + _processUnit->totalConnections();
}

size_t SparseSearchLayer::getFloatingPointOperationCount() const
{
    return _selectUnit->getFloatingPointOperationCount() +
        _processUnit->getFloatingPointOperationCount();
}

size_t SparseSearchLayer::getActivationMemory() const
{
    return _selectUnit->activationMemory() + _processUnit->activationMemory();
}

void SparseSearchLayer::save(util::OutputTarArchive& archive, util::PropertyTree& properties) const
{
    auto& selectUnit = properties["select-unit"];

    _selectUnit->save(archive, selectUnit);

    auto& processUnit = properties["process-unit"];

    _processUnit->save(archive, processUnit);

    properties["size"]       = _size;
    properties["depth"]      = _depth;
    properties["radix"]      = _radix;
    properties["selections"] = _selections;

    saveLayer(archive, properties);
}

void SparseSearchLayer::load(util::InputTarArchive& archive, const util::PropertyTree& properties)
{
    auto& selectUnit = properties["select-unit"];

    _selectUnit->load(archive, selectUnit);

    auto& processUnit = properties["process-unit"];

    _processUnit->load(archive, processUnit);

    _size       = properties["size"];
    _depth      = properties["depth"];
    _radix      = properties["radix"];
    _selections = properties["selections"];

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> SparseSearchLayer::clone() const
{
    return std::make_unique<Layer>(new SparseSearchLayer(*this));
}

std::string SparseSearchLayer::getTypeName() const
{
    return "SparseSearchLayer";
}

}

}






