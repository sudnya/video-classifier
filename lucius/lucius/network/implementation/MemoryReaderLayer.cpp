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
#include <lucius/matrix/interface/MatrixVectorOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/SoftmaxOperations.h>

#include <lucius/util/interface/debug.h>
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

    size_t aggregateInputSize = cellSize * (1 + cellCount);

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

    bundle["inputActivations"]  = MatrixVector({packedInputs});
    bundle["outputActivations"] = MatrixVector();

    return bundle;
}

static void unpackOutputsAndPerformWrite(
    Matrix& controllerOutput, Matrix& readOutput,
    const Matrix& memory, const Matrix& inputActivations,
    Bundle& bundle, size_t timestep)
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

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " softmax read address: "
            << readAddress.debugString();
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
        previousMemorySlice = inputActivations;
    }

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " previous contents: "
            << previousMemorySlice.debugString();
    }

    auto readOutputSlice = slice(readOutput,
        {0, 0, timestep}, {cellSize, miniBatchSize, timestep + 1});

    reduce(readOutputSlice, broadcast(previousMemorySlice, readAddress, {0}, matrix::Multiply()),
        {1}, matrix::Add());

    // Compute next timestep memory
    auto writeEnable = reshape(slice(controllerOutput,
        {0,             0, 0},
        {1, miniBatchSize, 1}), {miniBatchSize, 1});

    apply(writeEnable, writeEnable, matrix::Sigmoid());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " sigmoid write enable: "
            << writeEnable.debugString();
    }

    auto writeAddress = slice(controllerOutput,
        {1,                         0, 0},
        {1 + cellCount, miniBatchSize, 1});

    softmax(writeAddress, writeAddress);

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " softmax write address: "
            << writeAddress.debugString();
    }

    auto writeContents = slice(controllerOutput,
        {            1 + cellCount,             0, 0},
        {controllerWriteOutputSize, miniBatchSize, 1});

    Matrix newContents({cellSize, cellCount, miniBatchSize, 1}, controllerOutput.precision());

    broadcast(newContents, newContents, writeContents, {1},    matrix::CopyRight());
    broadcast(newContents, newContents, writeAddress,  {0},    matrix::Multiply());
    broadcast(newContents, newContents, writeEnable,   {0, 1}, matrix::Multiply());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " new contents: "
            << newContents.debugString();
    }

    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));
    auto scaledPreviousMemory = broadcast(previousMemorySlice, oneMinusWriteAddress, {0},
        matrix::Multiply());

    auto memorySlice = slice(memory,
        {       0,         0,             0, timestep},
        {cellSize, cellCount, miniBatchSize, timestep + 1});

    apply(memorySlice, scaledPreviousMemory, newContents, matrix::Add());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " current contents: "
            << memorySlice.debugString();
    }
}

static size_t getOutputTimesteps(const Bundle& bundle)
{
    if(bundle.contains("referenceActivations"))
    {
        auto& reference = bundle["referenceActivations"].get<MatrixVector>();

        return reference.back().size().back();
    }

    return 1;
}

void MemoryReaderLayer::runForwardImplementation(Bundle& bundle)
{
    auto&  inputActivationsVector = bundle[ "inputActivations"].get<MatrixVector>();
    auto& outputActivationsVector = bundle["outputActivations"].get<MatrixVector>();

    assert(inputActivationsVector.size() == 1);

    auto inputActivations = inputActivationsVector.back();

    util::log("MemoryReaderLayer") << " Running forward propagation of matrix "
        << inputActivations.shapeString() << "\n";

    size_t timesteps     = getOutputTimesteps(bundle);
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
        util::log("MemoryReaderLayer") << " Running forward propagation through controller "
            "on timestep " << timestep << "\n";
        getController().runForward(bundle);
        unpackOutputsAndPerformWrite(controllerOutput, output, memory, inputActivations,
            bundle, timestep);

        saveMatrix("controllerOutput", controllerOutput);
    }

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " outputs: " << output.debugString();
    }

    saveMatrix("inputActivations", inputActivations);
    saveMatrix("memory", memory);
    saveMatrix("output", output);

    outputActivationsVector.push_back(output);
}

static Bundle reversePropagationThroughWrite(Matrix& memory,
    Matrix& previousMemoryDeltas, const Matrix& controllerOutput,
    const Matrix& outputDeltas,
    const Matrix& currentMemoryDeltas, const Matrix& inputActivations,
    size_t timestep)
{
    size_t cellSize      = memory.size()[0];
    size_t cellCount     = memory.size()[1];
    size_t miniBatchSize = memory.size()[2];

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " output deltas: "
            << outputDeltas.debugString();
    }

    Matrix controllerOutputDeltas(controllerOutput.size(), controllerOutput.precision());

    // output is [writeScale, writeAddress, writeContents, readAddress]

    auto writeScale = reshape(slice(controllerOutput,
        {0,             0, 0},
        {1, miniBatchSize, 1}), {miniBatchSize, 1});
    auto writeScaleDeltas = reshape(slice(controllerOutputDeltas,
        {0,             0, 0},
        {1, miniBatchSize, 1}), {miniBatchSize, 1});

    auto writeAddress = slice(controllerOutput,
        {1            ,             0, 0},
        {1 + cellCount, miniBatchSize, 1});
    auto writeAddressDeltas = slice(controllerOutputDeltas,
        {1           ,              0, 0},
        {1 + cellCount, miniBatchSize, 1});

    auto writeContents = slice(controllerOutput,
        {1 + cellCount           ,             0, 0},
        {1 + cellCount + cellSize, miniBatchSize, 1});
    auto writeContentsDeltas = slice(controllerOutputDeltas,
        {1 + cellCount           ,             0, 0},
        {1 + cellCount + cellSize, miniBatchSize, 1});

    auto readAddress = slice(controllerOutput,
        {1 +     cellCount + cellSize,             0, 0},
        {1 + 2 * cellCount + cellSize, miniBatchSize, 1});
    auto readAddressDeltas = slice(controllerOutputDeltas,
        {1 +     cellCount + cellSize,             0, 0},
        {1 + 2 * cellCount + cellSize, miniBatchSize, 1});

    // Compute previousMemoryDeltas
    auto outputDeltasSlice = slice(outputDeltas,
        {0, 0, timestep}, {cellSize, miniBatchSize, timestep + 1});

    auto oneMinusWriteAddress = apply(writeAddress, matrix::Subtract(1.0));

    previousMemoryDeltas = broadcast(currentMemoryDeltas, oneMinusWriteAddress, {0},
        matrix::Multiply());

    Matrix readSoftmaxOutputDeltas({cellSize, cellCount, miniBatchSize, 1}, memory.precision());

    broadcast(readSoftmaxOutputDeltas, readSoftmaxOutputDeltas, readAddress,  {0},
        matrix::CopyRight());
    broadcast(readSoftmaxOutputDeltas, readSoftmaxOutputDeltas, outputDeltasSlice, {1},
        matrix::Multiply());

    apply(previousMemoryDeltas, previousMemoryDeltas, readSoftmaxOutputDeltas,
        matrix::Add());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " previous memory deltas: "
            << previousMemoryDeltas.debugString();
    }

    // Compute writeEnableDeltas
    Matrix contentsSoftmaxProduct({cellSize, cellCount, miniBatchSize, 1}, memory.precision());

    broadcast(contentsSoftmaxProduct, contentsSoftmaxProduct, writeAddress, {0},
        matrix::CopyRight());
    broadcast(contentsSoftmaxProduct, contentsSoftmaxProduct, writeContents, {1},
        matrix::Multiply());
    apply(contentsSoftmaxProduct, contentsSoftmaxProduct, currentMemoryDeltas, matrix::Multiply());

    apply(writeScaleDeltas, reduce(contentsSoftmaxProduct, {0, 1}, matrix::Add()),
        apply(writeScale, matrix::SigmoidDerivative()), matrix::Multiply());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " write enable deltas: "
            << writeScaleDeltas.debugString();
    }

    // Compute writeContentsDeltas
    auto currentScaledMemoryDeltas = broadcast(currentMemoryDeltas, writeScale,
        {0, 1}, matrix::Multiply());

    auto currentScaleAndAddressMemoryDeltas = broadcast(currentScaledMemoryDeltas,
        writeAddress, {0}, matrix::Multiply());

    reduce(writeContentsDeltas, currentScaleAndAddressMemoryDeltas, {1}, matrix::Add());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " write contents deltas: "
            << writeContentsDeltas.debugString();
    }

    // Compute writeAddressDeltas
    Matrix memorySlice;

    if(timestep == 0)
    {
        memorySlice = inputActivations;
    }
    else
    {
        memorySlice = slice(memory,
            {       0,         0,             0, timestep - 1},
            {cellSize, cellCount, miniBatchSize, timestep});
    }

    auto negativeMemoryAndCurrentMemoryDeltasProduct = apply(
        apply(Matrix(memorySlice), currentMemoryDeltas, matrix::Multiply()),
        matrix::Multiply(-1.0));

    auto currentScaledMemoryDeltasAndContentProduct =
        broadcast(currentScaledMemoryDeltas, writeContents, {1}, matrix::Multiply());

    auto softmaxWriteOutputGradient =
        reduce(apply(Matrix(negativeMemoryAndCurrentMemoryDeltasProduct),
            currentScaledMemoryDeltasAndContentProduct, matrix::Add()), {0}, matrix::Add());

    softmaxGradient(writeAddressDeltas, writeAddress, softmaxWriteOutputGradient);

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " write address deltas: "
            << writeAddressDeltas.debugString();
    }

    // Compute readAddressDeltas
    auto softmaxReadOutputGradient = reduce(
        broadcast(memorySlice, outputDeltasSlice, {1}, matrix::Multiply()),
        {0}, matrix::Add());

    softmaxGradient(readAddressDeltas, readAddress, softmaxReadOutputGradient);

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " read address deltas: "
            << readAddressDeltas.debugString();
    }

    Bundle bundle;

    bundle["outputDeltas"] = MatrixVector({controllerOutputDeltas});
    bundle["gradients"]    = MatrixVector();
    bundle["inputDeltas"]  = MatrixVector();

    return bundle;
}

static void unpackInputDeltas(Matrix& outputDeltas,
    Matrix& previousTimestepOutputMemoryDeltas,
    const Matrix& previousMemoryDeltas, const Bundle& bundle, size_t timestep)
{
    size_t cellSize      = previousMemoryDeltas.size()[0];
    size_t cellCount     = previousMemoryDeltas.size()[1];
    size_t miniBatchSize = previousMemoryDeltas.size()[2];

    // Compute previous output deltas
    auto& inputDeltasVector = bundle["inputDeltas"].get<MatrixVector>();

    // input deltas format is [memory size, cellSize]
    auto currentDeltasMemorySlice = reshape(slice(inputDeltasVector.back(),
        {                   0,             0, 0},
        {cellSize * cellCount, miniBatchSize, 1}), {cellSize, cellCount, miniBatchSize, 1});

    // Compute previous memory output deltas
    apply(previousTimestepOutputMemoryDeltas, previousMemoryDeltas, currentDeltasMemorySlice,
        matrix::Add());

    if(util::isLogEnabled("MemoryReaderLayer::Detail"))
    {
        util::log("MemoryReaderLayer::Detail") << " previous timestep memory deltas: "
            << previousTimestepOutputMemoryDeltas.debugString();
    }

    // Compute previous output deltas
    if(timestep > 0)
    {
        auto currentDeltasOutputSlice = reshape(
            slice(inputDeltasVector.back(),
                {           cellSize * cellCount,             0, 0},
                {cellSize * cellCount + cellSize, miniBatchSize, 1}),
            {cellSize, miniBatchSize, 1});

        auto previousTimestepOutputDeltas = slice(outputDeltas,
            {0, 0, timestep - 1}, {cellSize, miniBatchSize, timestep});

        apply(previousTimestepOutputDeltas, previousTimestepOutputDeltas, currentDeltasOutputSlice,
            matrix::Add());

        if(util::isLogEnabled("MemoryReaderLayer::Detail"))
        {
            util::log("MemoryReaderLayer::Detail") << " previous timestep output deltas: "
                << previousTimestepOutputDeltas.debugString();
        }
    }
}

void MemoryReaderLayer::runReverseImplementation(Bundle& bundle)
{
    auto& gradients          = bundle[   "gradients"].get<MatrixVector>();
    auto& inputDeltasVector  = bundle[ "inputDeltas"].get<MatrixVector>();
    auto& outputDeltasVector = bundle["outputDeltas"].get<MatrixVector>();

    auto memory = loadMatrix("memory");
    auto inputActivations = loadMatrix("inputActivations");

    size_t timesteps     = memory.size()[memory.size().size() - 1];
    size_t miniBatchSize = memory.size()[memory.size().size() - 2];

    // Get the output deltas
    auto outputDeltas = outputDeltasVector.front();

    Matrix outputMemoryDeltas = zeros({_cellSize, _cellCount, miniBatchSize, 1},
        outputDeltas.precision());

    for(size_t i = 0; i < timesteps; ++i)
    {
        size_t timestep = timesteps - i - 1;

        auto controllerOutput = loadMatrix("controllerOutput");
        popReversePropagationData();

        Matrix previousMemoryDeltas;
        auto controllerBundle = reversePropagationThroughWrite(memory, previousMemoryDeltas,
            controllerOutput, outputDeltas, outputMemoryDeltas,
            inputActivations, timestep);
        util::log("MemoryReaderLayer") << " Running reverse propagation through controller on "
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

        unpackInputDeltas(outputDeltas, outputMemoryDeltas, previousMemoryDeltas,
            controllerBundle, timestep);
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

