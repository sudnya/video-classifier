
// Lucius Includes
#include <lucius/matrix/interface/RecurrentOperations.h>

#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

namespace lucius
{
namespace matrix
{

void forwardRecurrentActivations(Matrix& input, const Matrix& weights,
    const RecurrentTimeDirection& direction, const Operation& activationFunction)
{
    bool reversed = (direction == RECURRENT_REVERSE_TIME);

    size_t timesteps     = input.size()[2];
    size_t miniBatchSize = input.size()[1];
    size_t layerSize     = input.size()[0];

    size_t currentTimestep = reversed ? timesteps - 1 : 0;

    // Start value
    auto currentInput = slice(input, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentInput, currentInput, activationFunction);

    // Propagate through time
    for(size_t timestep = 1; timestep < timesteps; ++timestep)
    {
        currentTimestep = reversed ? timesteps - timestep - 1 : timestep;

        auto nextInput = slice(input, {0, 0, currentTimestep},
            {layerSize, miniBatchSize, currentTimestep + 1});

        auto reshapedNextInput    = reshape(nextInput,    {layerSize, miniBatchSize});
        auto reshapedCurrentInput = reshape(currentInput, {layerSize, miniBatchSize});

        gemm(
            reshapedNextInput,           1.0,
            weights,              false, 1.0,
            reshapedCurrentInput, false);

        currentInput = nextInput;

        apply(currentInput, currentInput, activationFunction);
    }
}

Matrix forwardRecurrentActivations(const Matrix& input, const Matrix& weights,
    const RecurrentTimeDirection& d, const Operation& activationFunction)
{
    Matrix result = copy(input);

    forwardRecurrentActivations(result, weights, d, activationFunction);

    return result;
}

void reverseRecurrentDeltas(Matrix& resultAndInputDeltas, const Matrix& weights,
    const Matrix& activations, const RecurrentTimeDirection& direction,
    const Operation& activationDerivativeFunction)
{
    bool reversed = (direction == RECURRENT_REVERSE_TIME);

    size_t maxTimesteps     = resultAndInputDeltas.size()[2];
    size_t miniBatchSize    = resultAndInputDeltas.size()[1];
    size_t layerSize        = resultAndInputDeltas.size()[0];

    auto currentTimestep = reversed ? 0 : maxTimesteps - 1;
    // Start value
    auto currentDeltas = slice(resultAndInputDeltas, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});
    auto currentActivations = slice(activations, {0, 0, currentTimestep},
        {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentDeltas, currentDeltas,
        apply(currentActivations, activationDerivativeFunction), matrix::Multiply());

    //go over all timesteps in reverse
    for(size_t t = 1; t < maxTimesteps; ++t)
    {
        size_t timestep = reversed ? t : maxTimesteps - t - 1;

        auto previousDeltas = slice(resultAndInputDeltas, {0, 0, timestep},
            {layerSize, miniBatchSize, timestep + 1});

        auto reshapedPreviousDeltas = reshape(previousDeltas, {layerSize, miniBatchSize});
        auto reshapedCurrentDeltas  = reshape(currentDeltas,  {layerSize, miniBatchSize});

        gemm(
            reshapedPreviousDeltas, 1.0,
            weights, true, 1.0,
            reshapedCurrentDeltas, false
        );

        currentDeltas = previousDeltas;

        currentActivations = slice(activations, {0, 0, timestep},
            {layerSize, miniBatchSize, timestep + 1});

        apply(currentDeltas, currentDeltas,
            apply(currentActivations, activationDerivativeFunction), matrix::Multiply());
    }

}

Matrix reverseRecurrentDeltas(const Matrix& deltas, const Matrix& weights,
    const Matrix& activations, const RecurrentTimeDirection& d,
    const Operation& activationDerivativeFunction)
{
    Matrix result = copy(deltas);
    reverseRecurrentDeltas(result, weights, activations, d, activationDerivativeFunction);
    return result;
}

void reverseRecurrentGradients(Matrix& gradients, const Matrix& activations, const Matrix& deltas,
    const RecurrentTimeDirection& direction)
{
    bool reversed = (direction == RECURRENT_REVERSE_TIME);

    size_t maxTimesteps  = activations.size()[2];
    size_t miniBatchSize = activations.size()[1];
    size_t layerSize     = activations.size()[0];

    size_t deltaStart = reversed ?                0 : 1;
    size_t deltaEnd   = reversed ? maxTimesteps - 1 : maxTimesteps;

    size_t activationStart = reversed ?            1 : 0;
    size_t activationEnd   = reversed ? maxTimesteps : maxTimesteps - 1;

    auto currentDeltas = slice(deltas, {0, 0, deltaStart},
        {layerSize, miniBatchSize, deltaEnd});
    auto currentActivations = slice(activations, {0, 0, activationStart},
        {layerSize, miniBatchSize, activationEnd});

    gemm(
        gradients, 0.0,
        reshape(currentDeltas, {layerSize, miniBatchSize * (maxTimesteps - 1)}),
        false, (1.0 / miniBatchSize),
        reshape(currentActivations, {layerSize, miniBatchSize * (maxTimesteps - 1)} ),
        true
    );
}

Matrix reverseRecurrentGradients(const Matrix& inputs, const Matrix& deltas,
    const RecurrentTimeDirection& d)
{
    Matrix result({deltas.size()[0], inputs.size()[0]}, inputs.precision());

    reverseRecurrentGradients(result, inputs, deltas, d);

    return result;
}

void recurrentZeroEnds(Matrix& activations, const IndexVector& lengths)
{
    size_t maxTimesteps  = activations.size()[2];
    size_t miniBatchSize = activations.size()[1];
    size_t layerSize     = activations.size()[0];

    assert(miniBatchSize == lengths.size());

    for(size_t miniBatch = 0; miniBatch != miniBatchSize; ++miniBatch)
    {
        if(maxTimesteps == lengths[miniBatch])
        {
            continue;
        }

        auto activationsSlice = slice(activations, {0, miniBatch, lengths[miniBatch]},
            {layerSize, miniBatch + 1, maxTimesteps});

        zeros(activationsSlice);
    }
}

}
}



