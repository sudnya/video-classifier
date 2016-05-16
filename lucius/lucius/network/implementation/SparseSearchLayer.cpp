/*  \file   SparseSearchLayer.cpp
    \author Gregory Diamos
    \date   January 20, 2016
    \brief  The interface file for the SparseSearchLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/SparseSearchLayer.h>

#include <lucius/network/interface/SubgraphLayer.h>
#include <lucius/network/interface/LayerFactory.h>

#include <lucius/engine/interface/InputOutputLearnerEngine.h>

#include <lucius/matrix/interface/Precision.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/MatrixVector.h>

#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{
namespace network
{

typedef matrix::MatrixVector MatrixVector;
typedef matrix::Matrix       Matrix;
typedef matrix::Dimension    Dimension;

typedef engine::InputOutputLearnerEngine InputOutputLearnerEngine;

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

        layer->addLayer(layerName.str(), LayerFactory::create("FeedForwardLayer",
            util::ParameterPack(std::make_tuple("InputSize", inputSize),
            std::make_tuple("OutputSize", outputSize))));
    }

    return layer;
}

static size_t computeProcessedSize(size_t size, size_t selections)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeGeneratedSize(size_t size, size_t radix, size_t selections)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeProcessedTimeSize(size_t size)
{
    return size / 3;
}

static size_t computeSelectToProcessSize(size_t size)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeSelectInputSize(size_t size)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeProcessInputSize(size_t size)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeSelectCommandSize(size_t size)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeUnitSize(size_t radix, size_t size)
{
    assertM(false, "Not implemented.");

    return 0;
}

static size_t computeUnitCount(size_t radix, size_t size)
{
    assertM(false, "Not implemented.");

    return 0;
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

        layer->addLayer(layerName.str(), LayerFactory::create("FeedForwardLayer",
            util::ParameterPack(std::make_tuple("InputSize", inputSize),
            std::make_tuple("OutputSize", outputSize))));
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

static std::unique_ptr<SubgraphLayer> convert(std::unique_ptr<Layer>&& layer)
{
    auto* result = dynamic_cast<SubgraphLayer*>(layer.get());

    assert(result != nullptr);

    layer.release();

    return std::unique_ptr<SubgraphLayer>(result);
}

SparseSearchLayer::SparseSearchLayer(const SparseSearchLayer& l)
: Layer(l),
  _selectUnit(convert(l._selectUnit->clone())),
  _processUnit(convert(l._processUnit->clone())),
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

    _selectUnit  = convert(l._selectUnit->clone());
    _processUnit = convert(l._processUnit->clone());

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
    auto extendedSize = inputActivations.front().size();
    extendedSize[0]   = computeSelectInputSize(size);

    Matrix extendedInput(extendedSize, inputActivations.back().precision());

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
        auto previousTimestepActivation = zeros(inputActivation.size(),
            inputActivation.precision());

        copy(previousTimestepSlice, previousTimestepActivation);
        copy(currentTimestepSlice,  inputActivation);
    }

    MatrixVector result;
    MatrixVector extendedInputActivations;

    extendedInputActivations.push_back(extendedInput);

    layer.runForwardImplementation(result, extendedInputActivations);

    return result;
}

class RecursionState
{
public:
    RecursionState(SubgraphLayer& selectUnit, SubgraphLayer& processUnit,
        InputOutputLearnerEngine& engine, size_t size, size_t radix)
    : selectUnit(selectUnit),
      processUnit(processUnit),
      engine(engine),
      size(size),
      radix(radix)
    {

    }

public:
    SubgraphLayer& selectUnit;
    SubgraphLayer& processUnit;

    InputOutputLearnerEngine& engine;

    size_t size;
    size_t radix;

public:
    size_t segmentSize;
    size_t selectedSegment;
};

typedef std::vector<size_t> SegmentList;

static SegmentList computeSelectedSegments(const RecursionState& recursionState,
    const MatrixVector& selectResult)
{
    SegmentList result;

    assertM(false, "Not implemented.");

    return result;
}

static bool isLastRecursionLevel(const RecursionState& recursionState,
    const InputOutputLearnerEngine& engine)
{
    assertM(false, "Not implemented.");

    return true;
}

static size_t reduceSegmentSize(size_t originalSegmentSize)
{
    assertM(false, "Not implemented.");

    return 0;
}

static void runForward(MatrixVector& outputActivations,
    RecursionState& recursionState,
    const MatrixVector& inputActivations);

static MatrixVector runRecursionForward(RecursionState& recursionState,
    const MatrixVector& selectResult)
{
    // select based on the select unit's output
    auto selectedSegments = computeSelectedSegments(recursionState, selectResult);

    if(isLastRecursionLevel(recursionState, recursionState.engine))
    {
        size_t unitSize = computeUnitSize(recursionState.radix, recursionState.size);

        // read data directly from the engine
        Matrix resultData({recursionState.radix * unitSize, 1, 1},
            selectResult.front().precision());

        size_t index = 0;
        for(auto& selectedSegment : selectedSegments)
        {
            auto resultSlice = slice(resultData, {index * unitSize, 0, 0},
                {(index + 1) * unitSize, 1, 1});

            auto loadedData = recursionState.engine.getSlice(
                {selectedSegment}, {selectedSegment + unitSize});

            copy(resultSlice, loadedData);

            ++index;
        }

        MatrixVector result;

        result.push_back(resultData);

        return result;
    }
    else
    {
        size_t originalSegmentSize = recursionState.segmentSize;
        size_t newSegmentSize      = reduceSegmentSize(originalSegmentSize);

        recursionState.segmentSize = newSegmentSize;

        MatrixVector result;

        // recurse on the selected elements
        for(auto& selectedSegment : selectedSegments)
        {
            recursionState.selectedSegment = selectedSegment;

            MatrixVector input;

            runForward(result, recursionState, input);
        }

        recursionState.segmentSize = originalSegmentSize;

        return result;
    }
}

static Matrix sliceOutSelections(const Matrix& selectResult)
{
    Matrix result;

    assertM(false, "Not implemented.");

    return result;
}

static MatrixVector runRouterForward(const MatrixVector& selectResults,
    const MatrixVector& dataForSelectedUnits, size_t radix, size_t size)
{
    auto data         = dataForSelectedUnits.front();
    auto selectResult = sliceOutSelections(selectResults.front());

    size_t unitSize = computeUnitSize(radix, size);
    size_t units    = computeUnitCount(radix, size);

    auto reshapedData = reshape(data, {unitSize, units, 1, 1});

    MatrixVector result;

    result.push_back(broadcast(selectResult, reshapedData, {}, matrix::Multiply()));

    return result;
}

static MatrixVector runProcessForward(SubgraphLayer& layer, const MatrixVector& selectResult,
    const MatrixVector& generatedData, const MatrixVector& inputActivations, size_t size)
{
    auto extendedSize = inputActivations.front().size();
    extendedSize[0]   = computeProcessInputSize(size);

    Matrix extendedInput(extendedSize, inputActivations.front().precision());

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
        auto previousTimestepActivation = zeros(inputActivation.size(),
            inputActivation.precision());

        copy(previousTimestepSlice, previousTimestepActivation);
        copy(currentTimestepSlice,  inputActivation);
    }

    MatrixVector result;
    MatrixVector extendedInputActivations;

    extendedInputActivations.push_back(extendedInput);

    layer.runForward(result, extendedInputActivations);

    return result;
}

static void runForward(MatrixVector& outputActivations,
    RecursionState& recursionState,
    const MatrixVector& inputActivations)
{
    // Evaluate the select unit
    auto selectResult = runSelectForward(recursionState.selectUnit, inputActivations,
        recursionState.size);

    // Recurse based on the selected units
    auto dataForSelectedUnits = runRecursionForward(recursionState, selectResult);

    // Evaluate the router unit
    auto generatedData = runRouterForward(selectResult, dataForSelectedUnits,
        recursionState.radix, recursionState.size);

    // Evaluate the process unit
    auto processedData = runProcessForward(recursionState.processUnit, selectResult, generatedData,
        inputActivations, recursionState.size);

    outputActivations.push_back(std::move(processedData));
}

void SparseSearchLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivations  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivations = bundle["outputActivations"].get<MatrixVector>();

    RecursionState recursionState(*_selectUnit, *_processUnit, *_engine, _radix, _size);

    network::runForward(outputActivations, recursionState, inputActivations);
}

static void packOutputDeltas(MatrixVector& packedOutputDeltas, const MatrixVector& outputDeltas)
{
    assertM(false, "Not implemented.");
}

static MatrixVector runProcessReverse(SubgraphLayer& layer,
    MatrixVector& gradients, const MatrixVector& outputDeltas)
{
    MatrixVector generatedDataDeltas;
    MatrixVector packedOutputDeltas;

    packOutputDeltas(packedOutputDeltas, outputDeltas);

    layer.runReverse(gradients, generatedDataDeltas, packedOutputDeltas);

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

static MatrixVector runRouterReverse(MatrixVector& gradients,
    const MatrixVector& generatedDataDeltas)
{
    assertM(false, "Not Implemented");

    MatrixVector result;

    return result;
}

void SparseSearchLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients    = bundle[   "gradients"].get<MatrixVector>();
    auto& inputDeltas  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltas = bundle["outputDeltas"].get<MatrixVector>();

    // Evaluate the process unit
    auto generatedDataDeltas = runProcessReverse(*_processUnit, gradients, outputDeltas);

    // Evaluate the router unit
    auto selectedAndGeneratedDeltas = runRouterReverse(gradients, generatedDataDeltas);

    // Unwind the recursion
    auto selectDeltas = runRecursionReverse(*this, gradients, generatedDataDeltas);

    // Evaluate the select unit
    auto selectInputDeltas = runSelectReverse(gradients, generatedDataDeltas,
        selectedAndGeneratedDeltas);

    inputDeltas.push_back(selectInputDeltas);
}

MatrixVector& SparseSearchLayer::weights()
{
    return *_parameters;
}

const MatrixVector& SparseSearchLayer::weights() const
{
    return *_parameters;
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
    return {computeSelectCommandSize(_size), 1, 1};
}

Dimension SparseSearchLayer::getOutputSize() const
{
    return {computeProcessedSize(_size, _radix), 1, 1};
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
    return _selectUnit->getActivationMemory() + _processUnit->getActivationMemory();
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

    _size       = properties.get<size_t>("size");
    _depth      = properties.get<size_t>("depth");
    _radix      = properties.get<size_t>("radix");
    _selections = properties.get<size_t>("selections");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> SparseSearchLayer::clone() const
{
    return std::unique_ptr<Layer>(new SparseSearchLayer(*this));
}

std::string SparseSearchLayer::getTypeName() const
{
    return "SparseSearchLayer";
}

}

}






