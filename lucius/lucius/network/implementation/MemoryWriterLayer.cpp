/*  \file   MemoryWriterLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the MemoryWriterLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/MemoryWriterLayer.h>

#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/PropertyTree.h>

namespace lucius
{
namespace network
{

typedef matrix::Matrix       Matrix;
typedef matrix::Dimension    Dimension;
typedef matrix::MatrixVector MatrixVector;

MemoryWriterLayer::MemoryWriterLayer()
: MemoryWriterLayer(1, 1)
{

}

MemoryWriterLayer::MemoryWriterLayer(size_t cellSize, size_t cellCount, const Layer& controller)
: _cellSize(cellSize),
  _cellCount(cellCount),
  _controller(controller.clone())
{

}

MemoryWriterLayer::~MemoryWriterLayer()
{

}

MemoryWriterLayer::MemoryWriterLayer(const MemoryWriterLayer& l)
: Layer(l),
  _cellSize( l._cellSize),
  _cellCount(l._cellCount),
  _controller(l._controller->clone())
{

}

MemoryWriterLayer& MemoryWriterLayer::operator=(const MemoryWriterLayer& l)
{
    Layer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _cellSize  = l._cellSize;
    _cellCount = l._cellCount;

    _controller = l._controller->clone();

    return *this;
}

void MemoryWriterLayer::initialize()
{
    _controller->initialize();
}

static Bundle packInputsForWrite(const Matrix& memory, const Matrix& inputs, size_t timestep)
{
    size_t inputSize     = inputs.size()[0];
    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    size_t aggregateInputSize = inputSize * cellSize * cellCount;

    auto inputsSlice = slice(inputs,
        {        0,             0, timestep    },
        {inputSize, miniBatchSize, timestep + 1});

    Matrix packedInputs({aggregateInputSize, miniBatchSize, 1}, memory.precision());

    if(timestep != 0)
    {
        auto memorySlice = slice(memory,
            {       0,         0,             0, timestep - 1},
            {cellSize, cellCount, miniBatchSize, timestep    });

        auto packedInputsMemorySlice = slice(packedInputs,
            {         inputSize,             0, 0},
            {aggregateInputSize, miniBatchSize, 1});

        copy(packedInputsMemorySlice, memorySlice);
    }
    else
    {
        zeros(packedInputsMemorySlice);
    }

    auto packedInputsInputsSlice = slice(packedInputs,
        {        0,             0, 0,},
        {inputSize, miniBatchSize, 1});

    copy(packedInputsInputsSlice, inputsSlice);

    Bundle bundle;

    bundle["inputActivations"] = MatrixVector(packedInputs);

    return bundle;
}

static void unpackOutputsAndPerformWrite(
    Matrix& controllerOutput,
    const Matrix& memory, Bundle& bundle, size_t timestep)
{
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    controllerOutput = copy(outputActivationsVector.back());

    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    size_t controllerOutputSize = controllerOutput.size()[0];

    // output is [write enable, write address, write contents]
    assert(1 + cellCount + cellSize == controllerOutputSize);

    auto writeEnable = slice(controllerOutput,
        {0,             0, 0},
        {1, miniBatchSize, 1});

    apply(writeEnable, writeEnable, matrix::Sigmoid());

    auto writeAddress = slice(controllerOutput,
        {1,                         0, 0},
        {1 + cellCount, miniBatchSize, 1});

    softmax(writeAddress, writeAddress);

    auto writeContents = slice(controllerOutput,
        {       1 + cellCount,             0, 0},
        {controllerOutputSize, miniBatchSize, 1});

    Matrix newContents({cellSize, cellCount, miniBatchSize}, controllerOutput.precision());

    broadcast(newContents, newContents, writeContents, {1},    matrix::CopyRight());
    broadcast(newContents, newContents, writeAddress,  {0},    matrix::Multiply());
    broadcast(newContents, newContents, writeEnable,   {0, 1}, matrix::Multiply());

    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));
    auto scaledPreviousMemory = broadcast(previousMemorySlice, oneMinusWriteAddress, {0},
        matrix::Multiply());

    apply(memorySlice, scaledPreviousMemory, newContents, matrix::Add());
}

void MemoryWriterLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    util::log("MemoryWriterLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << "\n";

    auto inputActivations = inputActivationsVector.back();

    size_t timesteps     = inputActivations.size()[inputActivations.size().size() - 1];
    size_t miniBatchSize = inputActivations.size()[inputActivations.size().size() - 2];

    Matrix memory({_cellSize, _cellCount, miniBatchSize, timesteps}, inputActivations.precision());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " input: " << inputActivations.debugString();
    }

    for(size_t timestep = 0; timestep < timesteps; ++timestep)
    {
        Matrix controllerOutput;

        auto bundle = packInputsForWrite(memory, controllerOutput, inputActivations, timestep);
        _controller->runForward(bundle);
        unpackOutputsAndPerformWrite(memory, bundle, timestep);

        saveMatrix("controllerOutput", controllerOutput);
    }

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " outputs: " << outputActivations.debugString();
    }

    saveMatrix("memory", memory);

    auto finalMemoryState = slice(memory,
        {0,         0,          0,             timesteps - 1},
        {_cellSize, _cellCount, miniBatchSize, timesteps});

    outputActivationsVector.push_back(finalMemoryState);
}

static Bundle reversePropagationThroughWrite(Matrix& memory,
    Matrix& previousMemoryDeltas, const Matrix& controllerOutput,
    const Matrix& currentMemoryDeltas, size_t timestep)
{
    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    Matrix controllerOutputDeltas(controllerOutput.size(), controllerOutput.precision());

    // output is [writeScale, writeAddress, writeContents]

    auto writeScale = slice(controllerOutput,
        {0,             0},
        {1, miniBatchSize});
    auto writeScaleDeltas = slice(controllerOutputDeltas,
        {0,             0},
        {1, miniBatchSize});

    auto writeAddress = slice(controllerOutput,
        {            1,             0},
        {1 + cellCount, miniBatchSize});
    auto writeAddressDeltas = slice(controllerOutputDeltas,
        {            1,             0},
        {1 + cellCount, miniBatchSize});

    auto writeContents = slice(controllerOutput,
        {           1 + cellCount,             0},
        {1 + cellCount + cellSize, miniBatchSize});
    auto writeContentsDeltas = slice(controllerOutputDeltas,
        {           1 + cellCount,             0},
        {1 + cellCount + cellSize, miniBatchSize});

    // Compute previousMemoryDeltas
    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));

    previousMemoryDeltas = broadcast(currentMemoryDeltas, oneMinusWriteAddress, {0},
        matrix::Multiply());

    // Compute writeEnableDeltas
    Matrix contentsSoftmaxProduct(memory.size(), memory.precision());

    broadcast(contentsSoftmaxProduct, contentsSoftmaxProduct, writeAddress, {0},
        matrix::CopyRight());
    broadcast(contentsSoftmaxProduct, contentsSoftmaxProduct, writeContents, {1},
        matrix::Multiply());

    apply(writeEnableDeltas, reduce(contentsSoftmaxProduct, {0, 1}, matrix::Add()),
        matrix::SigmoidDerivative());

    // Compute writeContentsDeltas
    auto currentScaledMemoryDeltas = broadcast(currentMemoryDeltas, writeScale,
        {0, 1}, matrix::Multiply());

    auto currentScaleAndAddressMemoryDeltas = broadcast(currentScaledMemoryDeltas,
        writeAddress, {0}, matrix::Multiply());

    reduce(writeContentsDeltas, currentScaleAndAddressMemoryDeltas, {1}, matrix::Add());

    // Compute writeAddressDeltas
    auto negativeMemoryAndCurrentMemoryDeltasProduct =
        apply(apply(memory, currentMemoryDeltas, matrix::Multiply()), matrix::Multiply(-1.0));

    auto currentScaledMemoryDeltasAndContentProduct =
        broadcast(currentScaledMemoryDeltas, writeContents, {1}, matrix::Multiply());

    apply(writeAddressDeltas,
        apply(currentScaledMemoryDeltasAndContentProduct,
            negativeMemoryAndCurrentMemoryDeltasProduct, matrix::Add()),
        matrix::SigmoidDerivative());

    Bundle bundle;

    bundle["outputDeltas"] = MatrixVector(controllerOutputDeltas);

    return bundle;
}

static void unpackInputDeltas(Matrix& outputDeltas,
    Matrix& inputDeltas, const Matrix& previousMemoryDeltas,
    const Bundle& bundle, size_t timestep)
{
    size_t totalInputSize = inputDeltas.size()[0];
    size_t miniBatchSize  = inputDeltas.size()[1];

    // Compute input deltas
    auto& inputDeltasVector = bundle["inputDeltas"].get<MatrixVector>();

    auto currentInputDeltasSlice = slice(inputDeltasVector.back(),
        {        0,             0, 0},
        {inputSize, miniBatchSize, 1});

    auto inputDeltasSlice = slice(inputDeltas,
        {        0,             0, timestep},
        {inputSize, miniBatchSize, timestep + 1});

    copy(inputDeltasSlice, currentInputDeltasSlice);

    // Compute previous memory output deltas
    auto memoryDeltasSlice = slice(inputDeltasVector.back(),
        {     inputSize,             0, 0},
        {totalInputSize, miniBatchSize, 1});

    outputDeltas = apply(previousMemoryDeltas, memoryDeltasSlice, matrix::Add());
}

void MemoryWriterLayer::runReverseImplementation(Bundle& bundle)
{
    auto& inputDeltasVector  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

    // Get the output activations
    auto outputActivations = loadMatrix("outputActivations");

    // Get the output deltas
    auto outputDeltas = outputDeltasVector.front();

    Matrix inputDeltas({inputSize, miniBatchSize, timesteps}, outputDeltas.precision());

    auto& memory = loadMatrix("memory");

    for(size_t i = 0; i < timesteps; ++i)
    {
        size_t timestep = timesteps - i - 1;

        auto controllerOutput = loadMatrix("controllerOutput");
        popReversePropagationData();

        Matrix previousMemoryDeltas;
        auto bundle = reversePropagationThroughWrite(memory, previousMemoryDeltas,
            controllerOutput, outputDeltas, timestep);
        _controller->runReverse(bundle);
        _controller->popReversePropagationData();
        unpackInputDeltas(outputDeltas, inputDeltas, previousMemoryDeltas, bundle, timestep);
    }

    inputDeltasVector.push_back(inputDeltas);
}

MatrixVector& MemoryWriterLayer::weights()
{
    return _controller->weights();
}

const MatrixVector& MemoryWriterLayer::weights() const
{
    return _controller->weights();
}

const matrix::Precision& MemoryWriterLayer::precision() const
{
    return _controller->precision();
}

double MemoryWriterLayer::computeWeightCost() const
{
    return _controller->computeWeightCost();
}

Dimension MemoryWriterLayer::getInputSize() const
{
    return {_cellSize, 1, 1};
}

Dimension MemoryWriterLayer::getOutputSize() const
{
    return {_cellSize, _cellCount, 1, 1};
}

size_t MemoryWriterLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t MemoryWriterLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t MemoryWriterLayer::totalNeurons() const
{
    return _controller->totalNeurons();
}

size_t MemoryWriterLayer::totalConnections() const
{
    return _controller->totalConnections();
}

size_t MemoryWriterLayer::getFloatingPointOperationCount() const
{
    return _controller->getFloatingPointOperationCount();
}

size_t MemoryWriterLayer::getActivationMemory() const
{
    return _controller->getActivationMemory();
}

void MemoryWriterLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["cell-size"]  = _cellSize;
    properties["cell-count"] = _cellCount;

    auto& controllerProperties = properties["controller"];

    _controller->save(archive, controllerProperties);

    saveLayer(archive, properties);
}

void MemoryWriterLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _cellSize  = properties.get<size_t>("cell-size");
    _cellCount = properties.get<size_t>("cell-count");

    auto& controllerProperties = properties["controller"];

    _controller->load(archive, controllerProperties);

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> MemoryWriterLayer::clone() const
{
    return std::make_unique<MemoryWriterLayer>(*this);
}

std::string MemoryWriterLayer::getTypeName() const
{
    return "MemoryWriterLayer";
}

}

}

