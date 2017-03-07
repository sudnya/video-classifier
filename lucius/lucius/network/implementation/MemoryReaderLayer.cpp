/*  \file   MemoryReaderLayer.cpp
    \author Gregory Diamos
    \date   September 23, 2015
    \brief  The interface for the MemoryReaderLayer class.
*/

// Lucius Includes
#include <lucius/network/interface/MemoryReaderLayer.h>

#include <lucius/network/interface/ActivationFunction.h>
#include <lucius/network/interface/Bundle.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
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

MemoryReaderLayer::MemoryReaderLayer()
: MemoryReaderLayer(1, 1)
{

}

MemoryReaderLayer::MemoryReaderLayer(size_t cellSize, size_t cellCount)
: _cellSize(cellSize),
  _cellCount(cellCount)
{

}

MemoryReaderLayer::~MemoryReaderLayer()
{

}

MemoryReaderLayer::MemoryReaderLayer(const MemoryReaderLayer& l)
: ControllerLayer(l),
  _cellSize( l._cellSize),
  _cellCount(l._cellCount)
{

}

MemoryReaderLayer& MemoryReaderLayer::operator=(const MemoryReaderLayer& l)
{
    ControllerLayer::operator=(l);

    if(&l == this)
    {
        return *this;
    }

    _cellSize  = l._cellSize;
    _cellCount = l._cellCount;

    return *this;
}

void MemoryReaderLayer::initialize()
{
    getController().initialize();
}

static Bundle packInputsForWrite(const Matrix& memory, const Matrix& outputs,
    const Matrix& inputs, size_t timestep)
{
    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    size_t aggregateInputSize = cellSize * cellCount + cellSize;

    // input format is [memory contents, previous output]
    Matrix packedInputs({aggregateInputSize, miniBatchSize, 1}, memory.precision());

    // Copy previous timestep memory contents
    auto packedInputsMemorySlice = slice(packedInputs,
        {                   0,             0, 0},
        {cellSize * cellCount, miniBatchSize, 1});

    if(timestep != 0)
    {
        auto memorySlice = slice(memory,
            {       0,         0,             0, timestep - 1},
            {cellSize, cellCount, miniBatchSize, timestep    });

        copy(packedInputsMemorySlice, memorySlice);
    }
    else
    {
        copy(packedInputsMemorySlice, inputs);
    }

    // Copy previous timestep outputs
    auto packedInputsInputsSlice = slice(packedInputs,
        {cellSize * cellCount,             0, 0},
        {  aggregateInputSize, miniBatchSize, 1});

    if(timestep != 0)
    {
        auto outputsSlice = slice(outputs,
            {       0,             0, timestep - 1},
            {cellSize, miniBatchSize, timestep});

        copy(packedInputsInputsSlice, outputsSlice);
    }
    else
    {
        zeros(packedInputsInputsSlice);
    }

    Bundle bundle;

    bundle["inputActivations"] = MatrixVector({packedInputs});

    return bundle;
}

static void unpackOutputsAndPerformWrite(
    Matrix& controllerOutput, Matrix& readOutput,
    const Matrix& memory, Bundle& bundle, size_t timestep)
{
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    controllerOutput = copy(outputActivationsVector.back());

    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    size_t controllerWriteOutputSize = 1 + cellCount + cellSize;

    size_t controllerOutputSize = controllerOutput.size()[0];

    // output is [write enable, write address, write contents, read address]
    assert(1 + 2 * cellCount + cellSize == controllerOutputSize);

    // Compute current timestep output
    auto readAddress = slice(controllerOutput,
        {controllerWriteOutputSize,             0, 0},
        {     controllerOutputSize, miniBatchSize, 1});

    softmax(readAddress, readAddress);

    Matrix previousMemorySlice;

    if(timestep == 0)
    {
        previousMemorySlice = slice(memory,
            {       0,         0,             0, timestep - 1},
            {cellSize, cellCount, miniBatchSize, timestep});
    }
    else
    {
        previousMemorySlice = zeros({cellSize, cellCount, miniBatchSize, 1}, memory.precision());
    }

    reduce(readOutput, broadcast(previousMemorySlice, readAddress, {0}, matrix::Multiply()),
        {1}, matrix::Add());

    // Compute next timestep memory
    auto writeEnable = slice(controllerOutput,
        {0,             0, 0},
        {1, miniBatchSize, 1});

    apply(writeEnable, writeEnable, matrix::Sigmoid());

    auto writeAddress = slice(controllerOutput,
        {1,                         0, 0},
        {1 + cellCount, miniBatchSize, 1});

    softmax(writeAddress, writeAddress);

    auto writeContents = slice(controllerOutput,
        {            1 + cellCount,             0, 0},
        {controllerWriteOutputSize, miniBatchSize, 1});

    Matrix newContents({cellSize, cellCount, miniBatchSize}, controllerOutput.precision());

    broadcast(newContents, newContents, writeContents, {1},    matrix::CopyRight());
    broadcast(newContents, newContents, writeAddress,  {0},    matrix::Multiply());
    broadcast(newContents, newContents, writeEnable,   {0, 1}, matrix::Multiply());

    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));
    auto scaledPreviousMemory = broadcast(previousMemorySlice, oneMinusWriteAddress, {0},
        matrix::Multiply());

    auto memorySlice = slice(memory,
        {       0,         0,             0, timestep - 1},
        {cellSize, cellCount, miniBatchSize, timestep});

    apply(memorySlice, scaledPreviousMemory, newContents, matrix::Add());
}

void MemoryReaderLayer::runForwardImplementation(Bundle& bundle)
{
    auto&  inputActivationsVector = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    util::log("MemoryReaderLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << "\n";

    size_t timesteps     = inputActivations.size()[inputActivations.size().size() - 1];
    size_t miniBatchSize = inputActivations.size()[inputActivations.size().size() - 2];

    Matrix memory({_cellSize, _cellCount, miniBatchSize, timesteps}, inputActivations.precision());
    Matrix output({_cellSize,             miniBatchSize, timesteps}, inputActivations.precision());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " input: " << inputActivations.debugString();
    }

    for(size_t timestep = 0; timestep < timesteps; ++timestep)
    {
        Matrix controllerOutput;

        auto bundle = packInputsForWrite(memory, output, inputActivations, timestep);
        getController().runForward(bundle);
        unpackOutputsAndPerformWrite(controllerOutput, output, memory, bundle, timestep);

        saveMatrix("controllerOutput", controllerOutput);
    }

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " outputs: " << output.debugString();
    }

    saveMatrix("memory", memory);
    saveMatrix("output", output);

    outputActivationsVector.push_back(output);
}

static Bundle reversePropagationThroughWrite(Matrix& memory,
    Matrix& previousMemoryDeltas, const Matrix& controllerOutput,
    const Matrix& outputDeltas, const Matrix& previousOutputDeltas,
    const Matrix& currentMemoryDeltas, size_t timestep)
{
    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    Matrix controllerOutputDeltas(controllerOutput.size(), controllerOutput.precision());

    // output is [writeScale, writeAddress, writeContents, readContents]

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

    apply(writeScaleDeltas, reduce(contentsSoftmaxProduct, {0, 1}, matrix::Add()),
        matrix::SigmoidDerivative());

    // Compute writeContentsDeltas
    auto currentScaledMemoryDeltas = broadcast(currentMemoryDeltas, writeScale,
        {0, 1}, matrix::Multiply());

    auto currentScaleAndAddressMemoryDeltas = broadcast(currentScaledMemoryDeltas,
        writeAddress, {0}, matrix::Multiply());

    reduce(writeContentsDeltas, currentScaleAndAddressMemoryDeltas, {1}, matrix::Add());

    // Compute writeAddressDeltas
    auto negativeMemoryAndCurrentMemoryDeltasProduct = apply(
        apply(Matrix(memory), currentMemoryDeltas, matrix::Multiply()),
        matrix::Multiply(-1.0));

    auto currentScaledMemoryDeltasAndContentProduct =
        broadcast(currentScaledMemoryDeltas, writeContents, {1}, matrix::Multiply());

    apply(writeAddressDeltas,
        apply(Matrix(currentScaledMemoryDeltasAndContentProduct),
            negativeMemoryAndCurrentMemoryDeltasProduct, matrix::Add()),
        matrix::SigmoidDerivative());

    Bundle bundle;

    bundle["outputDeltas"] = MatrixVector({controllerOutputDeltas});

    return bundle;
}

static void unpackInputDeltas(Matrix& previousTimestepOutputDeltas,
    Matrix& previousTimestepOutputMemoryDeltas,
    const Matrix& previousMemoryDeltas, const Bundle& bundle, size_t timestep)
{
    size_t cellSize      = previousMemoryDeltas.size()[0];
    size_t cellCount     = previousMemoryDeltas.size()[1];
    size_t miniBatchSize = previousMemoryDeltas.size()[2];

    // Compute previous output deltas
    auto& inputDeltasVector = bundle["inputDeltas"].get<MatrixVector>();

    // input deltas format is [memory size, cellSize]

    auto currentDeltasMemorySlice = slice(inputDeltasVector.back(),
        {                   0,             0, 0},
        {cellSize * cellCount, miniBatchSize, 1});

    copy(previousTimestepOutputMemoryDeltas, currentDeltasMemorySlice);

    // Compute previous memory output deltas
    auto currentDeltasOutputSlice = slice(inputDeltasVector.back(),
        {           cellSize * cellCount,             0, 0},
        {cellSize * cellCount + cellSize, miniBatchSize, 1});

    apply(previousTimestepOutputDeltas, previousMemoryDeltas,
        currentDeltasOutputSlice, matrix::Add());
}

void MemoryReaderLayer::runReverseImplementation(Bundle& bundle)
{
    auto& inputDeltasVector  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

    // Get the output activations
    auto outputActivations = loadMatrix("outputActivations");

    // Get the output deltas
    auto outputDeltas = outputDeltasVector.front();

    size_t timesteps     = outputActivations.size()[outputActivations.size().size() - 1];
    size_t miniBatchSize = outputActivations.size()[outputActivations.size().size() - 2];

    Matrix outputMemoryDeltas = zeros({_cellSize, _cellCount, miniBatchSize},
        outputDeltas.precision());

    auto memory = loadMatrix("memory");

    auto previousOutputDeltas = zeros({_cellSize, miniBatchSize, 1}, outputDeltas.precision());

    for(size_t i = 0; i < timesteps; ++i)
    {
        size_t timestep = timesteps - i - 1;

        auto controllerOutput = loadMatrix("controllerOutput");
        popReversePropagationData();

        Matrix previousMemoryDeltas;
        auto bundle = reversePropagationThroughWrite(memory, previousMemoryDeltas,
            controllerOutput, outputDeltas, previousOutputDeltas, outputMemoryDeltas, timestep);
        getController().runReverse(bundle);
        getController().popReversePropagationData();
        unpackInputDeltas(previousOutputDeltas, outputMemoryDeltas, previousMemoryDeltas,
            bundle, timestep);
    }

    auto inputMemoryDeltas = outputMemoryDeltas;

    inputDeltasVector.push_back(inputMemoryDeltas);
}

MatrixVector& MemoryReaderLayer::weights()
{
    return getController().weights();
}

const MatrixVector& MemoryReaderLayer::weights() const
{
    return getController().weights();
}

const matrix::Precision& MemoryReaderLayer::precision() const
{
    return getController().precision();
}

double MemoryReaderLayer::computeWeightCost() const
{
    return getController().computeWeightCost();
}

Dimension MemoryReaderLayer::getInputSize() const
{
    return {_cellSize, _cellCount, 1, 1};
}

Dimension MemoryReaderLayer::getOutputSize() const
{
    return {_cellSize, 1, 1};
}

size_t MemoryReaderLayer::getInputCount() const
{
    return getInputSize().product();
}

size_t MemoryReaderLayer::getOutputCount() const
{
    return getOutputSize().product();
}

size_t MemoryReaderLayer::totalNeurons() const
{
    return getController().totalNeurons();
}

size_t MemoryReaderLayer::totalConnections() const
{
    return getController().totalConnections();
}

size_t MemoryReaderLayer::getFloatingPointOperationCount() const
{
    return getController().getFloatingPointOperationCount();
}

size_t MemoryReaderLayer::getActivationMemory() const
{
    return getController().getActivationMemory();
}

void MemoryReaderLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["cell-size"]  = _cellSize;
    properties["cell-count"] = _cellCount;

    saveLayer(archive, properties);
}

void MemoryReaderLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _cellSize  = properties.get<size_t>("cell-size");
    _cellCount = properties.get<size_t>("cell-count");

    loadLayer(archive, properties);
}

std::unique_ptr<Layer> MemoryReaderLayer::clone() const
{
    return std::make_unique<MemoryReaderLayer>(*this);
}

std::string MemoryReaderLayer::getTypeName() const
{
    return "MemoryReaderLayer";
}

}

}

