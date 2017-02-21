
// Lucius Includes
#include <lucius/matrix/interface/NativeRecurrentOperations.h>

#include <lucius/matrix/interface/RecurrentOperations.h>
#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Precision.h>

#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <tuple>

namespace lucius
{

namespace matrix
{

static size_t getMultiplier(const RecurrentOpsHandle& handle)
{
    size_t multiplier = 1;

    if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        multiplier *= 3;
    }
    else if(handle.layerType == RECURRENT_LSTM_TYPE)
    {
        multiplier *= 4;
    }

    return multiplier;
}

static size_t getInputMultiplier(const RecurrentOpsHandle& handle)
{
    if(handle.inputMode == RECURRENT_LINEAR_INPUT)
    {
        return 2;
    }

    return 1;
}

static size_t getDirectionMultiplier(const RecurrentOpsHandle& handle)
{
    if(handle.direction == RECURRENT_BIDIRECTIONAL)
    {
        return 2;
    }

    return 1;
}

static size_t getReserveSections(const RecurrentOpsHandle& handle)
{
    size_t savedActivations = getMultiplier(handle) *
        (getInputMultiplier(handle) * handle.layers - 1);

    size_t savedDeltas = getMultiplier(handle) * handle.layers;

    return savedActivations + savedDeltas;
}

size_t nativeRNNGetTrainingReserveSize(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    return precision.size() * getReserveSections(handle) * getDirectionMultiplier(handle) *
        handle.layerSize * handle.miniBatchSize * handle.timesteps;
}

size_t nativeRNNGetWeightsSize(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    size_t multiplier = getMultiplier(handle) * getInputMultiplier(handle) *
        getDirectionMultiplier(handle);

    size_t linearMatrixSize = handle.layers * handle.layerSize * handle.layerSize * multiplier;
    size_t biasSize         = handle.layers * handle.layerSize * multiplier;

    return (linearMatrixSize + biasSize) * precision.size();
}

std::tuple<size_t, size_t> nativeRNNGetLinearLayerMatrixSizeAndOffset(
    const RecurrentOpsHandle& handle, const Precision& precision,
    size_t layer, size_t offsetInLayer)
{
    size_t matrixSize = handle.layerSize * handle.layerSize * precision.size();
    size_t biasSize   = handle.layerSize * precision.size();

    size_t multiplier = getMultiplier(handle) * getInputMultiplier(handle);

    assert(offsetInLayer < multiplier);

    size_t layerStep = (matrixSize + biasSize) * multiplier;

    size_t size   = matrixSize;
    size_t offset = layer * layerStep + offsetInLayer * matrixSize;

    return std::make_tuple(size, offset);
}

std::tuple<size_t, size_t> nativeRNNGetBiasLayerMatrixSizeAndOffset(
    const RecurrentOpsHandle& handle, const Precision& precision,
    size_t layer, size_t offsetInLayer)
{
    size_t matrixSize = handle.layerSize * handle.layerSize * precision.size();
    size_t biasSize   = handle.layerSize * precision.size();

    size_t multiplier = getMultiplier(handle) * getInputMultiplier(handle);

    assert(offsetInLayer < multiplier);

    size_t layerStep = (matrixSize + biasSize) * multiplier;
    size_t biasStep  = biasSize;

    size_t size   = biasSize;
    size_t offset = layer * layerStep + multiplier * matrixSize + biasStep * offsetInLayer;

    return std::make_tuple(size, offset);
}

static void simpleRNNForwardPropRecurrentLinear(matrix::Matrix& expandedOutputs,
                                                const matrix::Matrix& expandedInputs,
                                                const matrix::Matrix& weights,
                                                const matrix::Matrix& bias)
{
    auto outputs = reshape(expandedOutputs, {expandedOutputs.size()[0]});
    auto inputs  = reshape(expandedInputs,  {expandedInputs.size()[0] });

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward input:   "
            << inputs.debugString();
        util::log("NativeRecurrentOperations::Detail") << "  forward weights: "
            << weights.debugString();
        util::log("NativeRecurrentOperations::Detail") << "  forward bias:    "
            << bias.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward input shape: "
            << inputs.shapeString() << "\n";
    }

    gemm(outputs,       0.0,
         weights,                              false, 1.0,
         reshape(inputs, {inputs.size()[0]}),  false);

    broadcast(outputs, outputs, bias, {}, matrix::Add());

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward output: "
            << outputs.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward output shape: "
            << outputs.shapeString() << "\n";
    }
}

static void simpleRNNForwardPropRecurrentThroughTime(matrix::Matrix& output,
                                                     const matrix::Matrix& recurrentWeights,
                                                     const matrix::Matrix& recurrentBias,
                                                     bool reversed,
                                                     const matrix::Operation& activationFunction)
{

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  recurrent input:   "
            << output.debugString();
        util::log("NativeRecurrentOperations::Detail") << "  recurrent weights: "
            << recurrentWeights.debugString();
        util::log("NativeRecurrentOperations::Detail") << "  recurrent bias:    "
            << recurrentBias.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  recurrent input shape: "
            << output.shapeString() << "\n";
    }

    double scale = util::KnobDatabase::getKnobValue(
        "NativeRecurrentOperations::SkipConnectionScale", 0.5);

    size_t timesteps     = output.size()[2];
    size_t miniBatchSize = output.size()[1];
    size_t layerSize     = output.size()[0];

    size_t currentTimestep = reversed ? timesteps - 1 : 0;

    // Start value
    auto currentInput = slice(output, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentInput, currentInput, activationFunction);

    // Propagate through time
    for(size_t timestep = 1; timestep < timesteps; ++timestep)
    {
        currentTimestep = reversed ? timesteps - timestep - 1 : timestep;

        auto nextInput = slice(output, {0, 0, currentTimestep},
            {layerSize, miniBatchSize, currentTimestep + 1});

        auto reshapedNextInput    = reshape(nextInput,    {layerSize, miniBatchSize});
        auto reshapedCurrentInput = reshape(currentInput, {layerSize, miniBatchSize});

        gemm(
            reshapedNextInput,           1.0,
            recurrentWeights,     false, 1.0,
            reshapedCurrentInput, false);

        apply(nextInput, nextInput, apply(currentInput, matrix::Multiply(scale)), matrix::Add());
        broadcast(nextInput, nextInput, recurrentBias, {}, matrix::Add());

        currentInput = nextInput;

        apply(currentInput, currentInput, activationFunction);
    }

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  recurrent forward output: "
            << output.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  recurrent forward output shape: "
            << output.shapeString() << "\n";
    }
}

static matrix::Matrix getForwardInput(const matrix::Matrix& inputs, const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle, size_t layer, size_t offset)
{
    if(layer == 0)
    {
        return inputs;
    }

    auto reshapedReserve = reshape(reserve,
        {handle.layerSize * getDirectionMultiplier(handle),
         handle.miniBatchSize,
         handle.timesteps,
         getReserveSections(handle)});

    auto slicedReserve = slice(reshapedReserve,
        {      offset * handle.layerSize,                    0,                0, layer - 1},
        {(offset + 1) * handle.layerSize, handle.miniBatchSize, handle.timesteps, layer});

    return reshape(slicedReserve, {handle.layerSize, handle.miniBatchSize, handle.timesteps});
}

static matrix::Matrix getForwardOutput(const matrix::Matrix& outputs,
    const matrix::Matrix& reserve, const RecurrentOpsHandle& handle, size_t layer, size_t offset)
{
    if(layer == handle.layers - 1)
    {
        return slice(outputs,
            {offset * handle.layerSize,                           0,                0},
            {(offset + 1) * handle.layerSize,  handle.miniBatchSize, handle.timesteps});
    }

    auto reshapedReserve = reshape(reserve,
        {handle.layerSize * getDirectionMultiplier(handle),
         handle.miniBatchSize,
         handle.timesteps,
         getReserveSections(handle)});

    auto slicedReserve = slice(reshapedReserve,
        {      offset * handle.layerSize,                     0,                0, layer},
        {(offset + 1) * handle.layerSize,  handle.miniBatchSize, handle.timesteps, layer + 1});

    return reshape(slicedReserve, {handle.layerSize, handle.miniBatchSize, handle.timesteps});
}

static matrix::Matrix getForwardWeights(const matrix::Matrix& allWeights,
                                        const RecurrentOpsHandle& handle,
                                        size_t logicalLayer,
                                        size_t offsetInLayer)
{
    auto sizeAndOffset = nativeRNNGetLinearLayerMatrixSizeAndOffset(
        handle, allWeights.precision(), logicalLayer + handle.layers * (offsetInLayer), 0);

    size_t size   = std::get<0>(sizeAndOffset) / allWeights.precision().size();
    size_t offset = std::get<1>(sizeAndOffset) / allWeights.precision().size();

    auto slicedWeights = slice(allWeights, {offset}, {offset + size});

    return reshape(slicedWeights, {handle.layerSize, handle.layerSize});
}

static matrix::Matrix getForwardBias(const matrix::Matrix& allWeights,
                                     const RecurrentOpsHandle& handle,
                                     size_t logicalLayer,
                                     size_t offsetInLayer)
{
    auto sizeAndOffset = nativeRNNGetBiasLayerMatrixSizeAndOffset(
        handle, allWeights.precision(), logicalLayer + handle.layers * (offsetInLayer), 0);

    size_t size   = std::get<0>(sizeAndOffset) / allWeights.precision().size();
    size_t offset = std::get<1>(sizeAndOffset) / allWeights.precision().size();

    auto slicedWeights = slice(allWeights, {offset}, {offset + size});

    return reshape(slicedWeights, {handle.layerSize});
}

static matrix::Matrix getRecurrentWeights(const matrix::Matrix& allWeights,
                                          const RecurrentOpsHandle& handle,
                                          size_t logicalLayer,
                                          size_t offsetInLayer)
{
    auto sizeAndOffset = nativeRNNGetLinearLayerMatrixSizeAndOffset(
        handle, allWeights.precision(), logicalLayer + handle.layers * (offsetInLayer), 1);

    size_t size   = std::get<0>(sizeAndOffset) / allWeights.precision().size();
    size_t offset = std::get<1>(sizeAndOffset) / allWeights.precision().size();

    auto slicedWeights = slice(allWeights, {offset}, {offset + size});

    return reshape(slicedWeights, {handle.layerSize, handle.layerSize});
}

static matrix::Matrix getRecurrentBias(const matrix::Matrix& allWeights,
                                       const RecurrentOpsHandle& handle,
                                       size_t logicalLayer,
                                       size_t offsetInLayer)
{
    auto sizeAndOffset = nativeRNNGetBiasLayerMatrixSizeAndOffset(
        handle, allWeights.precision(), logicalLayer + handle.layers * (offsetInLayer), 1);

    size_t size   = std::get<0>(sizeAndOffset) / allWeights.precision().size();
    size_t offset = std::get<1>(sizeAndOffset) / allWeights.precision().size();

    auto slicedWeights = slice(allWeights, {offset}, {offset + size});

    return reshape(slicedWeights, {handle.layerSize});
}

static matrix::Operation getSimpleRNNActivationFunction(const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE)
    {
        return matrix::RectifiedLinear();
    }
    else
    {
        return matrix::Tanh();
    }
}

static void simpleRNNForwardPropRecurrent(matrix::Matrix& allOutputActivations,
                                   const matrix::Matrix& allInputActivations,
                                   matrix::Matrix& reserve,
                                   const matrix::Matrix& allWeights,
                                   const RecurrentOpsHandle& handle,
                                   bool reversed,
                                   size_t offset)
{
    for(size_t layer = 0; layer < handle.layers; ++layer)
    {
        auto forwardInput = getForwardInput(allInputActivations, reserve,
            handle, layer, offset);
        auto forwardOutput = getForwardOutput(allOutputActivations, reserve,
            handle, layer, offset);

        if(handle.inputMode == RECURRENT_LINEAR_INPUT)
        {
            auto forwardWeights = getForwardWeights(allWeights, handle, layer, offset);
            auto forwardBias    = getForwardBias(   allWeights, handle, layer, offset);

            simpleRNNForwardPropRecurrentLinear(forwardOutput,
                                                forwardInput,
                                                forwardWeights,
                                                forwardBias);
        }
        else
        {
            copy(forwardOutput, forwardInput);
        }

        auto output = forwardOutput;

        auto recurrentWeights = getRecurrentWeights(allWeights, handle, layer, offset);
        auto recurrentBias    = getRecurrentBias(   allWeights, handle, layer, offset);

        simpleRNNForwardPropRecurrentThroughTime(output, recurrentWeights,
            recurrentBias, reversed, getSimpleRNNActivationFunction(handle));
    }
}

static void simpleRNNForwardPropRecurrent(matrix::Matrix& outputActivations,
                                          const matrix::Matrix& inputActivations,
                                          matrix::Matrix& reserve,
                                          const matrix::Matrix& weights,
                                          const RecurrentOpsHandle& handle)
{
    if(handle.direction == RECURRENT_BIDIRECTIONAL)
    {
        simpleRNNForwardPropRecurrent(outputActivations,
                                      inputActivations,
                                      reserve,
                                      weights,
                                      handle,
                                      false,
                                      0);
        simpleRNNForwardPropRecurrent(outputActivations,
                                      inputActivations,
                                      reserve,
                                      weights,
                                      handle,
                                      true,
                                      1);
    }
    else if(handle.direction == RECURRENT_FORWARD)
    {
        simpleRNNForwardPropRecurrent(outputActivations,
                                      inputActivations,
                                      reserve,
                                      weights,
                                      handle,
                                      false,
                                      0);
    }
    else
    {
        simpleRNNForwardPropRecurrent(outputActivations,
                                      inputActivations,
                                      reserve,
                                      weights,
                                      handle,
                                      true,
                                      0);
    }
}

static void gruForwardPropRecurrent(matrix::Matrix& outputActivations,
                                   const matrix::Matrix& inputActivations,
                                   matrix::Matrix& reserve,
                                   const matrix::Matrix& weights,
                                   const RecurrentOpsHandle& handle)
{
    assertM(false, "Not implemented");
}

static void lstmForwardPropRecurrent(matrix::Matrix& outputActivations,
                                   const matrix::Matrix& inputActivations,
                                   matrix::Matrix& reserve,
                                   const matrix::Matrix& weights,
                                   const RecurrentOpsHandle& handle)
{
    assertM(false, "Not implemented");
}

void nativeRNNForwardPropRecurrent(matrix::Matrix& outputActivations,
                                   const matrix::Matrix& inputActivations,
                                   matrix::Matrix& reserve,
                                   const matrix::Matrix& weights,
                                   const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE ||
        handle.layerType == RECURRENT_SIMPLE_TANH_TYPE)
    {
        simpleRNNForwardPropRecurrent(outputActivations,
                                      inputActivations,
                                      reserve,
                                      weights,
                                      handle);
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        gruForwardPropRecurrent(outputActivations,
                                inputActivations,
                                reserve,
                                weights,
                                handle);
    }
    else
    {
        assert(handle.layerType == RECURRENT_LSTM_TYPE);

        lstmForwardPropRecurrent(outputActivations,
                                 inputActivations,
                                 reserve,
                                 weights,
                                 handle);
    }
}

static void simpleRNNBackPropRecurrentThroughTime(matrix::Matrix& recurrentDeltas,
                                                  const matrix::Matrix& outputActivations,
                                                  const matrix::Matrix& recurrentWeights,
                                                  bool reversed,
                                                  const matrix::Operation& activationDerivative)
{
    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  recurrent output deltas:   "
            << recurrentDeltas.debugString();
        util::log("NativeRecurrentOperations::Detail") << "  recurrent weights: "
            << recurrentWeights.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  recurrent output deltas shape: "
            << recurrentDeltas.shapeString() << "\n";
    }

    double scale = util::KnobDatabase::getKnobValue(
        "NativeRecurrentOperations::SkipConnectionScale", 0.5);

    size_t maxTimesteps  = recurrentDeltas.size()[2];
    size_t miniBatchSize = recurrentDeltas.size()[1];
    size_t layerSize     = recurrentDeltas.size()[0];

    auto currentTimestep = reversed ? 0 : maxTimesteps - 1;
    // Start value
    auto currentDeltas = slice(recurrentDeltas, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});
    auto currentActivations = slice(outputActivations, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentDeltas, currentDeltas,
        apply(currentActivations, activationDerivative), matrix::Multiply());

    //go over all timesteps in reverse
    for(size_t t = 1; t < maxTimesteps; ++t)
    {
        size_t timestep = reversed ? t : maxTimesteps - t - 1;

        auto previousDeltas = slice(recurrentDeltas, {0, 0, timestep},
            {layerSize, miniBatchSize, timestep + 1});

        auto reshapedPreviousDeltas = reshape(previousDeltas, {layerSize, miniBatchSize});
        auto reshapedCurrentDeltas  = reshape(currentDeltas,  {layerSize, miniBatchSize});

        gemm(
            reshapedPreviousDeltas, 1.0,
            recurrentWeights, true, 1.0,
            reshapedCurrentDeltas, false
        );

        apply(previousDeltas, previousDeltas,
            apply(currentDeltas, matrix::Multiply(scale)),
            matrix::Add());

        currentDeltas = previousDeltas;

        currentActivations = slice(outputActivations, {0, 0, timestep},
            {layerSize, miniBatchSize, timestep + 1});

        apply(currentDeltas, currentDeltas,
            apply(currentActivations, activationDerivative),
            matrix::Multiply());
    }

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  recurrent input deltas:   "
            << recurrentDeltas.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  recurrent input deltas shape: "
            << recurrentDeltas.shapeString() << "\n";
    }
}

static void simpleRNNBackPropDeltasRecurrentLinear(matrix::Matrix& expandedForwardInputDeltas,
                                                   const matrix::Matrix& expandedRecurrentDeltas,
                                                   const matrix::Matrix& forwardWeights)
{
    auto forwardInputDeltas = reshape(expandedForwardInputDeltas,
                                      {expandedForwardInputDeltas.size()[0]});

    auto recurrentDeltas = reshape(expandedRecurrentDeltas,
                                  {expandedRecurrentDeltas.size()[0] });

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward output deltas:   "
            << recurrentDeltas.debugString();
        util::log("NativeRecurrentOperations::Detail") << "  forward weights: "
            << forwardWeights.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward output deltas shape: "
            << recurrentDeltas.shapeString() << "\n";
    }

    gemm(forwardInputDeltas,       1.0,
         forwardWeights,     true, 1.0,
         recurrentDeltas,    false);

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward input deltas:   "
            << forwardInputDeltas.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward input deltas shape: "
            << forwardInputDeltas.shapeString() << "\n";
    }
}

static matrix::Matrix getInputDeltas(const matrix::Matrix& inputs, const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle, size_t layer, size_t offset)
{
    if(layer == 0)
    {
        return inputs;
    }

    auto reshapedReserve = reshape(reserve,
        {handle.layerSize * getDirectionMultiplier(handle),
         handle.miniBatchSize,
         handle.timesteps,
         getReserveSections(handle)});

    size_t position = layer - 1 + (getInputMultiplier(handle) * handle.layers - 1);

    auto slicedReserve = slice(reserve,
        {      offset * handle.layerSize,                    0,                0, position},
        {(offset + 1) * handle.layerSize, handle.miniBatchSize, handle.timesteps, position + 1});

    return reshape(slicedReserve, {handle.layerSize, handle.miniBatchSize, handle.timesteps});
}

static matrix::Matrix getOutputDeltas(const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle, size_t layer, size_t offset)
{
    auto reshapedReserve = reshape(reserve,
        {handle.layerSize * getDirectionMultiplier(handle),
         handle.miniBatchSize,
         handle.timesteps,
         getReserveSections(handle)});

    size_t position = layer + (getInputMultiplier(handle) * handle.layers - 1);

    auto slicedReserve = slice(reshapedReserve,
        {      offset * handle.layerSize,                     0,                0, position},
        {(offset + 1) * handle.layerSize,  handle.miniBatchSize, handle.timesteps, position + 1});

    return reshape(slicedReserve, {handle.layerSize, handle.miniBatchSize, handle.timesteps});
}

static matrix::Operation getSimpleRNNActivationDerivative(const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE)
    {
        return matrix::RectifiedLinearDerivative();
    }
    else
    {
        return matrix::TanhDerivative();
    }

}

static void simpleRNNBackPropDeltasRecurrent(matrix::Matrix& inputDeltas,
                                             const matrix::Matrix& allOutputDeltas,
                                             const matrix::Matrix& allWeights,
                                             const matrix::Matrix& allOutputActivations,
                                             matrix::Matrix& reserve,
                                             const RecurrentOpsHandle& handle,
                                             bool reversed,
                                             size_t offset)
{
    // Save the output deltas in the reserve
    auto reserveDeltas = getOutputDeltas(reserve, handle, handle.layers - 1, offset);

    auto outputDeltas  = slice(allOutputDeltas,
        {      offset * handle.layerSize,                    0,                0},
        {(offset + 1) * handle.layerSize, handle.miniBatchSize, handle.timesteps});

    copy(reserveDeltas, outputDeltas);

    for(size_t l = 0; l < handle.layers; ++l)
    {
        size_t layer = handle.layers - l - 1;

        auto recurrentWeights = getRecurrentWeights(allWeights, handle, layer, offset);

        auto recurrentDeltas = getOutputDeltas(reserve, handle, layer, offset);
        auto forwardOutput   = getForwardOutput(allOutputActivations, reserve,
            handle, layer, offset);

        simpleRNNBackPropRecurrentThroughTime(recurrentDeltas, forwardOutput, recurrentWeights,
            reversed, getSimpleRNNActivationDerivative(handle));

        auto forwardInputDeltas = getInputDeltas(inputDeltas, reserve, handle, layer, offset);

        if(handle.inputMode == RECURRENT_LINEAR_INPUT)
        {
            auto forwardWeights = getForwardWeights(allWeights, handle, layer, offset);

            simpleRNNBackPropDeltasRecurrentLinear(forwardInputDeltas,
                                                   recurrentDeltas,
                                                   forwardWeights);
        }
        else
        {
            apply(forwardInputDeltas, forwardInputDeltas, recurrentDeltas, matrix::Add());
        }
    }
}

static void simpleRNNBackPropDeltasRecurrent(matrix::Matrix& inputDeltas,
                                          const matrix::Matrix& outputDeltas,
                                          const matrix::Matrix& weights,
                                          const matrix::Matrix& outputActivations,
                                          matrix::Matrix& reserve,
                                          const RecurrentOpsHandle& handle)
{
    zeros(inputDeltas);

    if(handle.direction == RECURRENT_BIDIRECTIONAL)
    {
        simpleRNNBackPropDeltasRecurrent(inputDeltas,
                                         outputDeltas,
                                         weights,
                                         outputActivations,
                                         reserve,
                                         handle,
                                         false,
                                         0);

        simpleRNNBackPropDeltasRecurrent(inputDeltas,
                                         outputDeltas,
                                         weights,
                                         outputActivations,
                                         reserve,
                                         handle,
                                         true,
                                         1);
    }
    else if(handle.direction == RECURRENT_FORWARD)
    {
        simpleRNNBackPropDeltasRecurrent(inputDeltas,
                                         outputDeltas,
                                         weights,
                                         outputActivations,
                                         reserve,
                                         handle,
                                         false,
                                         0);
    }
    else
    {
        simpleRNNBackPropDeltasRecurrent(inputDeltas,
                                         outputDeltas,
                                         weights,
                                         outputActivations,
                                         reserve,
                                         handle,
                                         true,
                                         0);
    }
}

static void gruRNNBackPropDeltasRecurrent(matrix::Matrix& inputDeltas,
                                          const matrix::Matrix& outputDeltas,
                                          const matrix::Matrix& weights,
                                          const matrix::Matrix& outputActivations,
                                          matrix::Matrix& reserve,
                                          const RecurrentOpsHandle& handle)
{
    assertM(false, "Not implemented.");
}

static void lstmRNNBackPropDeltasRecurrent(matrix::Matrix& inputDeltas,
                                           const matrix::Matrix& outputDeltas,
                                           const matrix::Matrix& weights,
                                           const matrix::Matrix& outputActivations,
                                           matrix::Matrix& reserve,
                                           const RecurrentOpsHandle& handle)
{
    assertM(false, "Not implemented.");
}

void nativeRNNBackPropDeltasRecurrent(matrix::Matrix& inputDeltas,
                                      const matrix::Matrix& outputDeltas,
                                      const matrix::Matrix& weights,
                                      const matrix::Matrix& outputActivations,
                                      matrix::Matrix& reserve,
                                      const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE ||
        handle.layerType == RECURRENT_SIMPLE_TANH_TYPE)
    {
        simpleRNNBackPropDeltasRecurrent(inputDeltas,
                                         outputDeltas,
                                         weights,
                                         outputActivations,
                                         reserve,
                                         handle);
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        gruRNNBackPropDeltasRecurrent(inputDeltas,
                                      outputDeltas,
                                      weights,
                                      outputActivations,
                                      reserve,
                                      handle);
    }
    else
    {
        assert(handle.layerType == RECURRENT_LSTM_TYPE);

        lstmRNNBackPropDeltasRecurrent(inputDeltas,
                                       outputDeltas,
                                       weights,
                                       outputActivations,
                                       reserve,
                                       handle);
    }
}

static void simpleRNNBackPropGradientsLinear(matrix::Matrix& dWeights,
                                             matrix::Matrix& dBias,
                                             const matrix::Matrix& expandedOutputDeltas,
                                             const matrix::Matrix& expandedInputActivations)
{

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward output deltas: "
            << expandedOutputDeltas.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward output deltas size: "
            << expandedOutputDeltas.shapeString() << "\n";
    }

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward output: "
            << expandedInputActivations.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward output size: "
            << expandedInputActivations.shapeString() << "\n";
    }

    size_t miniBatchSize = expandedInputActivations.size()[1];

    auto outputDeltas = reshape(expandedOutputDeltas, {expandedOutputDeltas.size()[0]});

    auto inputActivations = reshape(expandedInputActivations,
                                    {expandedInputActivations.size()[0]});

    gemm(dWeights,                0.0,
         outputDeltas,     false, 1.0 / miniBatchSize,
         inputActivations, true);

    reduce(dBias, apply(outputDeltas, matrix::Divide(miniBatchSize)), {1}, matrix::Add());

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward weight gradients: "
            << dWeights.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward weight gradient size: "
            << dWeights.shapeString() << "\n";
    }

    if(util::isLogEnabled("NativeRecurrentOperations::Detail"))
    {
        util::log("NativeRecurrentOperations::Detail") << "  forward bias gradients: "
            << dBias.debugString();
    }
    else
    {
        util::log("NativeRecurrentOperations") << "  forward bias gradient size: "
            << dBias.shapeString() << "\n";
    }
}

static void simpleRNNBackPropGradientsThroughTime(matrix::Matrix& dWeights,
                                             matrix::Matrix& dBias,
                                             const matrix::Matrix& outputDeltas,
                                             const matrix::Matrix& inputActivations,
                                             bool reversed)
{
    size_t maxTimesteps  = inputActivations.size()[2];
    size_t miniBatchSize = inputActivations.size()[1];
    size_t layerSize     = inputActivations.size()[0];

    size_t deltaStart = reversed ?                0 : 1;
    size_t deltaEnd   = reversed ? maxTimesteps - 1 : maxTimesteps;

    size_t activationStart = reversed ?            1 : 0;
    size_t activationEnd   = reversed ? maxTimesteps : maxTimesteps - 1;

    auto expandedCurrentDeltas = slice(outputDeltas,
        {        0,             0, deltaStart},
        {layerSize, miniBatchSize, deltaEnd  });
    auto expandedCurrentActivations = slice(inputActivations,
        {        0,             0, activationStart},
        {layerSize, miniBatchSize, activationEnd  });

    auto currentDeltas = reshape(expandedCurrentDeltas,
                                 {layerSize, miniBatchSize * (maxTimesteps - 1)});

    auto currentActivations = reshape(expandedCurrentActivations,
                                      {layerSize, miniBatchSize * (maxTimesteps - 1)});

    gemm(
        dWeights, 0.0,
        currentDeltas,
        false, (1.0 / miniBatchSize),
        currentActivations,
        true
    );

    reduce(dBias, apply(currentDeltas, matrix::Divide(miniBatchSize)), {1}, matrix::Add());
}

static void simpleRNNBackPropGradientsRecurrent(matrix::Matrix& dWeights,
                                                const matrix::Matrix& allInputActivations,
                                                const matrix::Matrix& allOutputActivations,
                                                const matrix::Matrix& reserve,
                                                const RecurrentOpsHandle& handle,
                                                bool reversed,
                                                size_t offset)
{
    for(size_t layer = 0; layer < handle.layers; ++layer)
    {
        auto forwardInput = getForwardInput(allInputActivations, reserve,
            handle, layer, offset);
        auto forwardOutputDeltas = getOutputDeltas(reserve, handle, layer, offset);

        if(handle.inputMode == RECURRENT_LINEAR_INPUT)
        {
            auto forwardDWeights = getForwardWeights(dWeights, handle, layer, offset);
            auto forwardDBias    = getForwardBias(   dWeights, handle, layer, offset);

            simpleRNNBackPropGradientsLinear(forwardDWeights,
                                             forwardDBias,
                                             forwardOutputDeltas,
                                             forwardInput);
        }

        auto outputActivations = getForwardOutput(allOutputActivations, reserve,
            handle, layer, offset);

        auto recurrentOutputDeltas = getOutputDeltas(reserve, handle, layer, offset);

        auto recurrentDWeights = getRecurrentWeights(dWeights, handle, layer, offset);
        auto recurrentDBias    = getRecurrentBias(   dWeights, handle, layer, offset);

        simpleRNNBackPropGradientsThroughTime(recurrentDWeights, recurrentDBias,
            recurrentOutputDeltas, outputActivations, reversed);
    }
}

static void simpleRNNBackPropGradientsRecurrent(matrix::Matrix& dWeights,
                                                const matrix::Matrix& inputActivations,
                                                const matrix::Matrix& outputActivations,
                                                const matrix::Matrix& reserve,
                                                const RecurrentOpsHandle& handle)
{
    if(handle.direction == RECURRENT_BIDIRECTIONAL)
    {
        simpleRNNBackPropGradientsRecurrent(dWeights,
                                            inputActivations,
                                            outputActivations,
                                            reserve,
                                            handle,
                                            false,
                                            0);

        simpleRNNBackPropGradientsRecurrent(dWeights,
                                            inputActivations,
                                            outputActivations,
                                            reserve,
                                            handle,
                                            true,
                                            1);
    }
    else if(handle.direction == RECURRENT_FORWARD)
    {
        simpleRNNBackPropGradientsRecurrent(dWeights,
                                            inputActivations,
                                            outputActivations,
                                            reserve,
                                            handle,
                                            false,
                                            0);
    }
    else
    {
        simpleRNNBackPropGradientsRecurrent(dWeights,
                                            inputActivations,
                                            outputActivations,
                                            reserve,
                                            handle,
                                            true,
                                            0);
    }

}

static void gruRNNBackPropGradientsRecurrent(matrix::Matrix& dWeights,
                                             const matrix::Matrix& inputActivations,
                                             const matrix::Matrix& outputActivations,
                                             const matrix::Matrix& reserve,
                                             const RecurrentOpsHandle& handle)
{
    assertM(false, "Not implemented.");
}

static void lstmRNNBackPropGradientsRecurrent(matrix::Matrix& dWeights,
                                              const matrix::Matrix& inputActivations,
                                              const matrix::Matrix& outputActivations,
                                              const matrix::Matrix& reserve,
                                              const RecurrentOpsHandle& handle)
{
    assertM(false, "Not implemented.");
}

void nativeRNNBackPropGradientsRecurrent(matrix::Matrix& dWeights,
                                         const matrix::Matrix& inputActivations,
                                         const matrix::Matrix& outputActivations,
                                         const matrix::Matrix& reserve,
                                         const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE ||
        handle.layerType == RECURRENT_SIMPLE_TANH_TYPE)
    {
        return simpleRNNBackPropGradientsRecurrent(dWeights,
                                                   inputActivations,
                                                   outputActivations,
                                                   reserve,
                                                   handle);
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        return gruRNNBackPropGradientsRecurrent(dWeights,
                                                inputActivations,
                                                outputActivations,
                                                reserve,
                                                handle);
    }
    else
    {
        assert(handle.layerType == RECURRENT_LSTM_TYPE);

        return lstmRNNBackPropGradientsRecurrent(dWeights,
                                                 inputActivations,
                                                 outputActivations,
                                                 reserve,
                                                 handle);
    }
}

}

}

