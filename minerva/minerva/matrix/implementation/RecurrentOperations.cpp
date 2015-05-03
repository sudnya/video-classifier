
// Minerva Includes
#include <minerva/matrix/interface/RecurrentOperations.h>

#include <minerva/matrix/interface/BlasOperations.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixTransformations.h>

#include <minerva/parallel/interface/MultiBulkSynchronousParallel.h>

namespace minerva
{
namespace matrix
{

void forwardRecurrentActivations(Matrix& input, const Matrix& weights, const RecurrentTimeDirection& direction, const Operation& activationFunction)
{
    bool reversed = (direction == RECURRENT_REVERSE_TIME);

    size_t timesteps     = input.size()[2];
    size_t miniBatchSize = input.size()[1];
    size_t layerSize     = input.size()[0];

    size_t currentTimestep = reversed ? timesteps - 1 : 0;

    // Start value
    auto currentInput = slice(input, {0, 0, currentTimestep}, {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentInput, currentInput, activationFunction);

    // Propagate through time
    for(size_t timestep = 1; timestep < timesteps; ++timestep)
    {
        currentTimestep = reversed ? timesteps - timestep - 1 : timestep;

        auto nextInput = slice(input, {0, 0, currentTimestep}, {layerSize, miniBatchSize, currentTimestep + 1});
        /*
        wt    Current input
        [1 2 3]   [1 2]
        [4 6 7] * [2 1] => O/P (3,2)
        [4 5 1]   [4 5]

        O/P + nextInput
        */
        gemm(
            reshape(nextInput,    {layerSize, miniBatchSize}),        1.0,
            weights,                                           false, 1.0,
            reshape(currentInput, {layerSize, miniBatchSize}), false);

        currentInput = nextInput;

        apply(currentInput, currentInput, activationFunction);
    }
}

Matrix forwardRecurrentActivations(const Matrix& input, const Matrix& weights, const RecurrentDirection& d, const Operation& activationFunction)
{
    Matrix result = copy(input);

    forwardRecurrentActivations(result, weights, d, activationFunction);

    return result;
}

void reverseRecurrentDeltas(Matrix& resultAndInputDeltas, const Matrix& weights,
    const RecurrentDirection& direction, const Operation& activationDerivativeFunction)
{
    bool reversed = (direction == RECURRENT_REVERSE_TIME);

    size_t maxTimesteps     = resultAndInputDeltas.size()[2];
    size_t miniBatchSize    = resultAndInputDeltas.size()[1];
    size_t layerSize        = resultAndInputDeltas.size()[0];

    auto currentTimestep = reversed ? 0 : maxTimesteps - 1;
    // Start value
    auto currentDeltas = slice(resultAndInputDeltas, {0, 0, currentTimestep}, {layerSize, miniBatchSize, currentTimestep + 1});

    apply(currentDeltas, currentDeltas, activationDerivativeFunction);

    //go over all timesteps
    for(long timestep = maxTimesteps-2; timestep >= 0; --timestep)
    {
        auto previousDeltas = slice(resultAndInputDeltas, {0, 0, timestep}, {layerSize, miniBatchSize, timestep + 1});

        gemm(
            reshape(previousDeltas, {layerSize, miniBatchSize}), 1.0,
            weights, true, 1.0,
            reshape(currentDeltas,  {layerSize, miniBatchSize}), false
        );

        currentDeltas = previousDeltas;

        apply(currentDeltas, currentDeltas, activationDerivativeFunction);
    }

}

Matrix reverseRecurrentDeltas(const Matrix& weights, const Matrix& deltas, const RecurrentDirection& d,
    const Operation& activationDerivativeFunction)
{

    Matrix result = copy(deltas);
    reverseRecurrentDeltas(result, weights, d, activationDerivativeFunction);
    return result;
}

void reverseRecurrentGradients(Matrix& gradients, const Matrix& activations, const Matrix& deltas)
{
    size_t maxTimesteps  = activations.size()[2];
    size_t miniBatchSize = activations.size()[1];
    size_t layerSize     = activations.size()[0];


    auto currentDeltas = slice(deltas, {0, 0, 1}, {layerSize, miniBatchSize, maxTimesteps});

    auto currentActivations = slice(activations, {0, 0, 0}, {layerSize, miniBatchSize, maxTimesteps-1});

    gemm(
        gradients, 0.0,
        reshape(currentDeltas,      {layerSize, miniBatchSize*(maxTimesteps - 1)} ), false, (1.0/miniBatchSize),
        reshape(currentActivations, {layerSize, miniBatchSize*(maxTimesteps - 1)} ), true
    );
}

Matrix reverseRecurrentGradients(const Matrix& inputs, const Matrix& deltas)
{
    Matrix result({inputs.size()[0], inputs.size()[0]}, inputs.precision());

    reverseRecurrentGradients(result, inputs, deltas);

    return result;
}


}
}



