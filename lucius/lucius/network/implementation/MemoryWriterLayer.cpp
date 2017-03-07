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
#include <lucius/matrix/interface/MatrixVectorOperations.h>
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

MemoryWriterLayer::MemoryWriterLayer()
: MemoryWriterLayer(1, 1)
{

}

MemoryWriterLayer::MemoryWriterLayer(size_t cellSize, size_t cellCount)
: _cellSize(cellSize),
  _cellCount(cellCount)
{

}

MemoryWriterLayer::~MemoryWriterLayer()
{

}

MemoryWriterLayer::MemoryWriterLayer(const MemoryWriterLayer& l)
: ControllerLayer(l),
  _cellSize( l._cellSize),
  _cellCount(l._cellCount)
{

}

MemoryWriterLayer& MemoryWriterLayer::operator=(const MemoryWriterLayer& l)
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

void MemoryWriterLayer::initialize()
{
    getController().initialize();
}

static Bundle packInputsForWrite(const Matrix& memory, const Matrix& inputs, size_t timestep)
{
    size_t inputSize     = inputs.size()[0];
    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    size_t aggregateInputSize = cellSize * (1 + cellCount);

    auto inputsSlice = slice(inputs,
        {        0,             0, timestep    },
        {inputSize, miniBatchSize, timestep + 1});

    Matrix packedInputs({aggregateInputSize, miniBatchSize, 1}, memory.precision());

    auto packedInputsMemorySlice = slice(packedInputs,
        {         inputSize,             0, 0},
        {aggregateInputSize, miniBatchSize, 1});

    if(timestep != 0)
    {
        auto memorySlice = slice(memory,
            {       0,         0,             0, timestep - 1},
            {cellSize, cellCount, miniBatchSize, timestep    });

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

    bundle["inputActivations"]  = MatrixVector({packedInputs});
    bundle["outputActivations"] = MatrixVector();

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

    auto writeEnable = reshape(slice(controllerOutput,
        {0,             0, 0},
        {1, miniBatchSize, 1}), {miniBatchSize, 1});

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " write enable: " << writeEnable.debugString();
    }

    apply(writeEnable, writeEnable, matrix::Sigmoid());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " sigmoid write enable: "
            << writeEnable.debugString();
    }

    auto writeAddress = slice(controllerOutput,
        {1,                         0, 0},
        {1 + cellCount, miniBatchSize, 1});

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " write address: "
            << writeAddress.debugString();
    }

    softmax(writeAddress, writeAddress);

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " softmax write address: "
            << writeAddress.debugString();
    }

    auto writeContents = slice(controllerOutput,
        {       1 + cellCount,             0, 0},
        {controllerOutputSize, miniBatchSize, 1});

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " write contents: "
            << writeContents.debugString();
    }

    Matrix newContents({cellSize, cellCount, miniBatchSize, 1}, controllerOutput.precision());

    broadcast(newContents, newContents, writeContents, {1},    matrix::CopyRight());
    broadcast(newContents, newContents, writeAddress,  {0},    matrix::Multiply());
    broadcast(newContents, newContents, writeEnable,   {0, 1}, matrix::Multiply());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " new contents: "
            << newContents.debugString();
    }

    Matrix previousMemorySlice;

    if(timestep != 0)
    {
        previousMemorySlice = slice(memory,
            {       0,         0,             0, timestep - 1},
            {cellSize, cellCount, miniBatchSize, timestep});
    }
    else
    {
        previousMemorySlice = zeros({cellSize, cellCount, miniBatchSize, 1}, memory.precision());
    }

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " previous contents: "
            << previousMemorySlice.debugString();
    }

    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));
    auto scaledPreviousMemory = broadcast(previousMemorySlice, oneMinusWriteAddress, {0},
        matrix::Multiply());

    auto memorySlice = slice(memory,
        {       0,         0,             0, timestep},
        {cellSize, cellCount, miniBatchSize, timestep + 1});

    apply(memorySlice, scaledPreviousMemory, newContents, matrix::Add());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " current contents: "
            << memorySlice.debugString();
    }
}

void MemoryWriterLayer::runForwardImplementation(Bundle& bundle)
{
    auto& inputActivationsVector  = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    util::log("MemoryWriterLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << "\n";

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

        auto bundle = packInputsForWrite(memory, inputActivations, timestep);
        util::log("MemoryWriterLayer") << " Running forward propagation through controller "
            "on timestep " << timestep << "\n";
        getController().runForward(bundle);
        unpackOutputsAndPerformWrite(controllerOutput, memory, bundle, timestep);

        saveMatrix("controllerOutput", controllerOutput);
    }

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " outputs: " << memory.debugString();
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

    auto writeScale = reshape(slice(controllerOutput,
        {0,             0, 0},
        {1, miniBatchSize, 1}), {miniBatchSize, 1});
    auto writeScaleDeltas = reshape(slice(controllerOutputDeltas,
        {0,             0, 0},
        {1, miniBatchSize, 1}), {miniBatchSize, 1});

    auto writeAddress = slice(controllerOutput,
        {            1,             0, 0},
        {1 + cellCount, miniBatchSize, 1});
    auto writeAddressDeltas = slice(controllerOutputDeltas,
        {            1,             0, 0},
        {1 + cellCount, miniBatchSize, 1});

    auto writeContents = slice(controllerOutput,
        {           1 + cellCount,             0, 0},
        {1 + cellCount + cellSize, miniBatchSize, 1});
    auto writeContentsDeltas = slice(controllerOutputDeltas,
        {           1 + cellCount,             0, 0},
        {1 + cellCount + cellSize, miniBatchSize, 1});

    // Compute previousMemoryDeltas
    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));

    previousMemoryDeltas = broadcast(currentMemoryDeltas, oneMinusWriteAddress, {0},
        matrix::Multiply());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " previous memory deltas: "
            << previousMemoryDeltas.debugString();
    }

    // Compute writeEnableDeltas
    Matrix contentsSoftmaxProduct(memory.size(), memory.precision());

    broadcast(contentsSoftmaxProduct, contentsSoftmaxProduct, writeAddress, {0},
        matrix::CopyRight());
    broadcast(contentsSoftmaxProduct, contentsSoftmaxProduct, writeContents, {1},
        matrix::Multiply());
    apply(contentsSoftmaxProduct, contentsSoftmaxProduct, currentMemoryDeltas, matrix::Multiply());

    apply(writeScaleDeltas, reduce(contentsSoftmaxProduct, {0, 1}, matrix::Add()),
        apply(writeScale, matrix::SigmoidDerivative()), matrix::Multiply());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " write enable deltas: "
            << writeScaleDeltas.debugString();
    }

    // Compute writeContentsDeltas
    auto currentScaledMemoryDeltas = broadcast(currentMemoryDeltas, writeScale,
        {0, 1}, matrix::Multiply());

    auto currentScaleAndAddressMemoryDeltas = broadcast(currentScaledMemoryDeltas,
        writeAddress, {0}, matrix::Multiply());

    reduce(writeContentsDeltas, currentScaleAndAddressMemoryDeltas, {1}, matrix::Add());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " write contents deltas: "
            << writeContentsDeltas.debugString();
    }

    // Compute writeAddressDeltas
    auto negativeMemoryAndCurrentMemoryDeltasProduct = apply(
        apply(Matrix(memory), currentMemoryDeltas, matrix::Multiply()),
        matrix::Multiply(-1.0));

    auto currentScaledMemoryDeltasAndContentProduct =
        broadcast(currentScaledMemoryDeltas, writeContents, {1}, matrix::Multiply());

    apply(writeAddressDeltas,
        reduce(apply(Matrix(currentScaledMemoryDeltasAndContentProduct),
            negativeMemoryAndCurrentMemoryDeltasProduct, matrix::Add()), {1}, matrix::Add()),
        matrix::SigmoidDerivative());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " write address deltas: "
            << writeAddressDeltas.debugString();
    }

    Bundle bundle;

    bundle["outputDeltas"] = MatrixVector({controllerOutputDeltas});
    bundle["gradients"]    = MatrixVector();
    bundle["inputDeltas"]  = MatrixVector();

    return bundle;
}

static void unpackInputDeltas(Matrix& outputDeltas,
    Matrix& inputDeltas, const Matrix& previousMemoryDeltas,
    const Bundle& bundle, size_t timestep)
{
    // Compute input deltas
    auto& inputDeltasVector = bundle["inputDeltas"].get<MatrixVector>();

    size_t inputSize      = inputDeltas.size()[0];
    size_t totalInputSize = inputDeltasVector.back().size()[0];
    size_t miniBatchSize  = inputDeltasVector.back().size()[1];

    auto currentInputDeltasSlice = slice(inputDeltasVector.back(),
        {        0,             0, 0},
        {inputSize, miniBatchSize, 1});

    auto inputDeltasSlice = slice(inputDeltas,
        {        0,             0, timestep},
        {inputSize, miniBatchSize, timestep + 1});

    copy(inputDeltasSlice, currentInputDeltasSlice);

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " input deltas: "
            << inputDeltasSlice.debugString();
    }

    // Compute previous memory output deltas
    size_t cellSize  = previousMemoryDeltas.size()[0];
    size_t cellCount = previousMemoryDeltas.size()[1];

    auto memoryDeltasSlice = reshape(slice(inputDeltasVector.back(),
        {     inputSize,             0, 0},
        {totalInputSize, miniBatchSize, 1}), {cellSize, cellCount, miniBatchSize, 1});

    outputDeltas = apply(previousMemoryDeltas, memoryDeltasSlice, matrix::Add());

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " previous memory deltas: "
            << outputDeltas.debugString();
    }
}

void MemoryWriterLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients          = bundle[   "gradients"].get<MatrixVector>();
    auto& inputDeltasVector  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

    // Get the output deltas
    auto& outputDeltas = outputDeltasVector.front();

    util::log("MemoryWriterLayer") << " Running reverse propagation of matrix "
        << outputDeltas.shapeString() << "\n";

    if(util::isLogEnabled("MemoryWriterLayer::Detail"))
    {
        util::log("MemoryWriterLayer::Detail") << " output deltas: " << outputDeltas.debugString();
    }

    size_t timesteps     = outputDeltas.size()[outputDeltas.size().size() - 1];
    size_t miniBatchSize = outputDeltas.size()[outputDeltas.size().size() - 2];

    Matrix inputDeltas({_cellSize, miniBatchSize, timesteps}, outputDeltas.precision());

    auto memory = loadMatrix("memory");

    for(size_t i = 0; i < timesteps; ++i)
    {
        size_t timestep = timesteps - i - 1;

        auto controllerOutput = loadMatrix("controllerOutput");
        popReversePropagationData();

        Matrix previousMemoryDeltas;
        auto controllerBundle = reversePropagationThroughWrite(memory, previousMemoryDeltas,
            controllerOutput, outputDeltas, timestep);
        util::log("MemoryWriterLayer") << " Running reverse propagation through controller on "
            "timestep " << timestep << "\n";
        getController().runReverse(controllerBundle);
        getController().popReversePropagationData();

        auto& controllerGradients = controllerBundle["gradients"].get<MatrixVector>();

        if(i == 0)
        {
            gradients = controllerGradients;
        }
        else
        {
            apply(gradients, gradients, controllerGradients, matrix::Add());
        }

        unpackInputDeltas(outputDeltas, inputDeltas, previousMemoryDeltas,
            controllerBundle, timestep);
    }

    inputDeltasVector.push_back(inputDeltas);
}

MatrixVector& MemoryWriterLayer::weights()
{
    return getController().weights();
}

const MatrixVector& MemoryWriterLayer::weights() const
{
    return getController().weights();
}

const matrix::Precision& MemoryWriterLayer::precision() const
{
    return getController().precision();
}

double MemoryWriterLayer::computeWeightCost() const
{
    return getController().computeWeightCost();
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
    return getController().totalNeurons();
}

size_t MemoryWriterLayer::totalConnections() const
{
    return getController().totalConnections();
}

size_t MemoryWriterLayer::getFloatingPointOperationCount() const
{
    return getController().getFloatingPointOperationCount();
}

size_t MemoryWriterLayer::getActivationMemory() const
{
    return getController().getActivationMemory();
}

void MemoryWriterLayer::save(util::OutputTarArchive& archive,
    util::PropertyTree& properties) const
{
    properties["cell-size"]  = _cellSize;
    properties["cell-count"] = _cellCount;

    saveLayer(archive, properties);
}

void MemoryWriterLayer::load(util::InputTarArchive& archive,
    const util::PropertyTree& properties)
{
    _cellSize  = properties.get<size_t>("cell-size");
    _cellCount = properties.get<size_t>("cell-count");

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


