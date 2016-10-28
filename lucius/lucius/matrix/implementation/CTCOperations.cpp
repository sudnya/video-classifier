// Lucius Includes
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/CTCOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/parallel/interface/Synchronization.h>

#include <lucius/util/interface/debug.h>

// Forward Declarations
namespace lucius
{
namespace matrix
{

static void initializeWorkspace(Matrix& workspace, const Matrix& inputActivations, size_t beamSize)
{
    auto currentInputActivations = slice(inputActivations,
        {       0,             0, timestep},
        {alphabet, miniBatchSize, timestep + 1});

    // copy max input activation into all character positions
    auto max = reduce(currentInputActivations, {0}, Maximum());

    broadcast(workspace, workspace, max, CopyRight());
}

static void expandBeam(Matrix& workspace, const Matrix& inputActivations,
    size_t beamSize, size_t timestep)
{
    // multiply existing probability values for each path on the beam with the next character prob
    auto currentInputActivations = slice(inputActivations,
        {       0,             0, timestep},
        {alphabet, miniBatchSize, timestep + 1});

    currentInputActivations = reshape(currentInputActivations, {alphanet, miniBatchSize});

    broadcast(workspace, workspace, currentInputActivations, Multiply());
}

static void sortBeam(Matrix& workspace)
{
    Matrix augmentedWorkspace({2, alphabet, beamSize, miniBatchSize}, workspace.precision());

    auto keys = slice(augmentedWorkspace, {0, 0, 0, 0}, {1, alphabet, beamSize, miniBatchSize});

    copy(keys, workspace);

    auto path = slice(augmentedWorkspace, {1, 0, 0, 0}, {2, alphabet, beamSize, miniBatchSize});
    zero(path);
    broadcast(path, path, range({alphabetSize}, workspace.precision()), Add());

    sortByKey(keys, values, {0, 1, 2});

    return augmentedWorkspace;
}

static void pruneBeam(Matrix& workspace, Matrix& outputActivations, Matrix& inputPaths,
    const Matrix& augmentedWorkspace, size_t beamSize, size_t timestep)
{
    // Generate paths
    auto inputPathsSlice = slice(inputPaths,
        {       0,             0, timestep},
        {beamSize, miniBatchSize, timestep + 1});

    auto compressedAugmentedWorkspace = reshape(augmentedWorkspace,
        {2, beamSize * alphabet, miniBatchSize});

    auto augmentedWorkspacePaths = slice(compressedAugmentedWorkspace,
        {1,        0,             0},
        {2, beamSize, miniBatchSize});

    copy(inputPathsSlice, augmentedWorkspacePaths);

    // Generate one-hot outputs
    auto outputActivationSlice = slice(outputActivations,
        {       0,        0,             0, timestep},
        {alphabet, beamSize, miniBatchSize, timestep + 1});

    gather(outputActivationsSlice, augmentedWorkspacePaths, GatherIndexToOneHot({0})); // TODO

    // Fill in the workspace for the next step
    auto augmentedWorkspaceProbabilities = slice(compressedAugmentedWorkspace,
        {0,        0,             0},
        {1, beamSize, miniBatchSize});

    broadcast(workspace, workspace, augmentedWorkspaceProbabilities, CopyRight());
}

void ctcBeamSearch(Matrix& outputActivationWeights, Matrix& inputPaths, Matrix& outputActivations,
    const Matrix& inputActivations, size_t beamSize)
{
    size_t alphabet      = inputActivations.size().front();
    size_t miniBatchSize = inputActivations.size()[1];
    size_t timesteps     = inputActivations.size().back();

    Matrix workspace({alphabet, beamSize, miniBatchSize}, inputActivations.precision());

    initializeWorkspace(workspace, inputActivations, beamSize);

    for(size_t timestep = 0; timestep != timesteps; ++timestep)
    {
        expandBeam(workspace, inputActivations, beamSize, timestep);
        auto augmentedWorkspace = sortBeam(workspace);
        pruneBeam(workspace, outputActivations, inputPaths,
            augmentedWorkspace, beamSize, timestep);
    }
}

Matrix computeFactors(const Matrix& scanPositions)
{
    auto factors = reduceByKey(scanPositions,
        ones(scanPositions.size(), scanPositions.precision()), Add());

    return indirectGather(scanPositions, apply(factors, Divide(1.0)),
        MapOutputToIndexDimension({0, 1, 2}, {0}, {1, 2, 3}));
}

void ctcBeamSearchInputGradients(Matrix& inputDeltas, const Matrix& outputActivationWeights,
    const Matrix& inputPaths, const Matrix& outputDeltas, size_t beamSize)
{
    // {beamSize, miniBatchSize, timesteps}
    auto transitionPositions = applyToAdjacentElements(inputPaths, {2}, NotEqual(), 0.0);

    // {beamSize, miniBatchSize, timesteps}
    auto scanPositions = exclusiveScan(transitionPositions, {2}, Add());

    auto factors = computeFactors(scanPositions);

    // output deltas {alphabet, beamSize, miniBatchSize, timesteps}
    // scan positions {beamSize, miniBatchSize, timesteps}
    // selected output deltas size is {beamSize, miniBatchSize, timesteps}
    auto selectedOutputDeltas = indirectGather(outputDeltas, scanPositions,
        MapOutputToIndexDimension({0, 1, 2}, {0}, {1, 2, 3}));

    auto scaledOutputDeltas = apply(factors, selectedOutputDeltas, Multiply());

    // scaled output deltas size is {beamSize, miniBatchSize, timesteps}
    // input paths is {beamSize, miniBatchSize, timesteps}
    // expanded input deltas {alphabet, beamSize, miniBatchSize, timesteps}
    auto expandedInputDeltas = indirectGather(scaledOutputDeltas, inputPaths,
        MapOutputToMatchingIndexDimension({1, 2, 3}, {0}, {1, 2, 3}));

    reduce(inputDeltas, expandedInputDeltas, {1}, Add());
}

}
}


