// Lucius Includes
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/CTCOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/GatherOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/SortOperations.h>
#include <lucius/matrix/interface/ScanOperations.h>
#include <lucius/matrix/interface/ReduceByKeyOperations.h>
#include <lucius/matrix/interface/AdjacentElementOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/GatherOperation.h>

#include <lucius/parallel/interface/Synchronization.h>

#include <lucius/util/interface/debug.h>

// Forward Declarations
namespace lucius
{
namespace matrix
{

static void advancePaths(Matrix& inputPaths, const Matrix& workspace, size_t timestep)
{
    size_t beamSize      = workspace.size()[0];
    size_t alphabet      = workspace.size()[1];
    size_t miniBatchSize = workspace.size()[2];

    Matrix possiblePaths(workspace.size(), workspace.precision());

    broadcast(possiblePaths, possiblePaths, range({alphabet}, workspace.precision()),
        {0, 2}, CopyRight());

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  possible paths : "
            << possiblePaths.debugString();
    }

    auto sortedWorkspace = copy(workspace);
    sortByKey(sortedWorkspace, possiblePaths, {0, 1, 2}, GreaterThan());

    auto pathSlice = reshape(
        slice(inputPaths, {0, 0, timestep}, {beamSize, miniBatchSize, timestep + 1}),
        {beamSize, miniBatchSize});

    auto possiblePathsSlice = reshape(
        slice(possiblePaths, {0, 0, 0}, {beamSize, 1, miniBatchSize}),
        {beamSize, miniBatchSize});

    copy(pathSlice, possiblePathsSlice);

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  updated paths : "
            << inputPaths.debugString();
    }
}

static void sortWorkspace(Matrix& workspace)
{
    sort(workspace, {0, 1, 2}, GreaterThan());

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  sorted workspace : "
            << workspace.debugString();
    }
}

static void initializeWorkspace(Matrix& workspace, Matrix& inputPaths,
    const Matrix& inputActivations)
{
    size_t alphabet      = inputActivations.size()[0];
    size_t miniBatchSize = inputActivations.size()[1];

    auto currentInputActivations = reshape(
        slice(inputActivations,
            {       0,             0, 0},
            {alphabet, miniBatchSize, 1}),
        {alphabet, miniBatchSize});

    auto workspaceSlice = reshape(
        slice(workspace, {0, 0, 0}, {1, alphabet, miniBatchSize}),
        {alphabet, miniBatchSize});

    zeros(workspace);
    copy(workspaceSlice, currentInputActivations);

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  initial workspace : "
            << workspace.debugString();
    }

    advancePaths(inputPaths, workspace, 0);
    sortWorkspace(workspace);
}

Dimension getBeamSearchOutputSize(const Dimension& inputSize, size_t beamSize)
{
    size_t alphabet      = inputSize[0];
    size_t miniBatchSize = inputSize[1];
    size_t timesteps     = inputSize[2];

    return {alphabet, beamSize, miniBatchSize, timesteps};
}

static void expandBeam(Matrix& workspace, const Matrix& inputActivations,
    size_t timestep)
{
    size_t alphabet      = inputActivations.size()[0];
    size_t miniBatchSize = inputActivations.size()[1];

    size_t beamSize = workspace.size()[0];

    auto currentInputActivations = slice(inputActivations,
        {       0,             0, timestep},
        {alphabet, miniBatchSize, timestep + 1});

    currentInputActivations = reshape(currentInputActivations, {alphabet, miniBatchSize});

    auto workspaceSlice =
        reshape(slice(workspace, {0, 0, 0}, {beamSize, 1, miniBatchSize}),
            {beamSize, miniBatchSize});

    broadcast(workspace, workspace, copy(workspaceSlice), {1}, CopyRight());

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  pruned workspace : "
            << workspace.debugString();
    }

    broadcast(workspace, workspace, currentInputActivations, {0}, Multiply());

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  workspace for timestep " << timestep << " : "
            << workspace.debugString();
    }
}

static void advanceLinks(Matrix& links, const Matrix& workspace, size_t timestep)
{
    size_t beamSize      = workspace.size()[0];
    size_t miniBatchSize = workspace.size()[2];

    Matrix possibleLinks(workspace.size(), workspace.precision());

    broadcast(possibleLinks, possibleLinks, range({beamSize}, workspace.precision()),
        {1, 2}, CopyRight());

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  possible links : "
            << possibleLinks.debugString();
    }

    auto sortedWorkspace = copy(workspace);
    sortByKey(sortedWorkspace, possibleLinks, {0, 1, 2}, GreaterThan());

    auto linksSlice = reshape(
        slice(links, {0, 0, timestep - 1}, {beamSize, miniBatchSize, timestep}),
        {beamSize, miniBatchSize});

    auto possibleLinksSlice = reshape(
        slice(possibleLinks, {0, 0, 0}, {beamSize, 1, miniBatchSize}),
        {beamSize, miniBatchSize});

    copy(linksSlice, possibleLinksSlice);

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  updated links : "
            << links.debugString();
    }
}

static void advanceBeam(Matrix& workspace, Matrix& inputPaths, Matrix& links,
    const Matrix& inputActivations, size_t timestep)
{
    expandBeam(workspace, inputActivations, timestep);

    advancePaths(inputPaths, workspace, timestep);
    advanceLinks(links, workspace, timestep);

    sortWorkspace(workspace);

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  updated workspace: "
            << workspace.debugString();
    }
}

static void createFullPaths(Matrix& inputPaths, const Matrix& links)
{
    Matrix pathLookupTable(inputPaths.size(), inputPaths.precision());

    size_t beamSize      = inputPaths.size()[0];
    size_t miniBatchSize = inputPaths.size()[1];
    size_t timesteps     = inputPaths.size()[2];

    auto initialSlice = slice(pathLookupTable, {0, 0, timesteps - 1},
        {beamSize, miniBatchSize, timesteps});

    broadcast(initialSlice, initialSlice, range({beamSize}, inputPaths.precision()), {1, 2},
        CopyRight());

    for(size_t i = 0; i < timesteps - 1; ++i)
    {
        size_t currentTimestep = timesteps - i - 1;
        size_t nextTimestep    = timesteps - i - 2;

        auto nextSlice = slice(pathLookupTable, {0, 0, nextTimestep},
            {beamSize, miniBatchSize, nextTimestep + 1});
        auto nextLinks = slice(links, {0, 0, nextTimestep},
            {beamSize, miniBatchSize, nextTimestep + 1});
        auto currentSlice = slice(pathLookupTable, {0, 0, currentTimestep},
            {beamSize, miniBatchSize, currentTimestep + 1});

        indirectGather(nextSlice, currentSlice, nextLinks,
            MapOutputToIndexDimension({0, 1, 2}, 0, {1, 2}));
    }

    indirectGather(inputPaths, copy(inputPaths), pathLookupTable,
        MapOutputToIndexDimension({0, 1, 2}, 0, {1, 2}));

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  path lookup table: "
            << pathLookupTable.debugString();
        util::log("CTCOperations::Detail") << "  final paths: "
            << inputPaths.debugString();
    }
}

static void setOutputActivationWeights(Matrix& outputActivationWeights,
    const Matrix& workspace)
{
    size_t beamSize      = workspace.size()[0];
    size_t miniBatchSize = workspace.size()[2];

    auto workspaceSlice = reshape(
        slice(workspace, {0, 0, 0}, {beamSize, 1, miniBatchSize}),
        {beamSize, miniBatchSize});

    copy(outputActivationWeights, workspaceSlice);

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  output activation weights: "
            << outputActivationWeights.debugString();
    }
}

static void setOutputActivations(Matrix& outputActivations, const Matrix& inputPaths)
{

    size_t beamSize      = inputPaths.size()[0];
    size_t miniBatchSize = inputPaths.size()[1];
    size_t timesteps     = inputPaths.size()[2];

    auto reshapedInputPaths = reshape(inputPaths, {1, beamSize, miniBatchSize, timesteps});

    gather(outputActivations, reshapedInputPaths, GatherIndexToOneHot(0));

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  output activations: "
            << outputActivations.debugString();
    }
}

static void finalizeWorkspace(Matrix& outputActivationWeights, Matrix& outputActivations,
    Matrix& inputPaths, const Matrix& links, const Matrix& workspace)
{
    createFullPaths(inputPaths, links);
    setOutputActivationWeights(outputActivationWeights, workspace);
    setOutputActivations(outputActivations, inputPaths);
}

void ctcBeamSearch(Matrix& outputActivationWeights, Matrix& inputPaths,
    Matrix& outputActivations, const Matrix& inputActivations, size_t beamSize)
{
    size_t alphabet      = inputActivations.size().front();
    size_t miniBatchSize = inputActivations.size()[1];
    size_t timesteps     = inputActivations.size().back();

    Matrix workspace({beamSize, alphabet, miniBatchSize}, inputActivations.precision());
    Matrix links(inputPaths.size(), inputPaths.precision());

    initializeWorkspace(workspace, inputPaths, inputActivations);

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << "  input activations: "
            << inputActivations.debugString();
    }

    for(size_t timestep = 1; timestep != timesteps; ++timestep)
    {
        if(util::isLogEnabled("CTCOperations::Detail"))
        {
            util::log("CTCOperations::Detail") << " processing timestep: "
                << timestep << "\n";
        }

        advanceBeam(workspace, inputPaths, links, inputActivations, timestep);
    }

    finalizeWorkspace(outputActivationWeights, outputActivations, inputPaths,
        links, workspace);
}

Matrix computeFactors(const Matrix& scanPositions)
{
    auto factors = reduceByKey(scanPositions,
        ones(scanPositions.size(), scanPositions.precision()), {2}, Add());

    Matrix selectedFactors(scanPositions.size(), scanPositions.precision());

    indirectGather(selectedFactors, scanPositions, apply(factors, Divide(1.0)),
        MapOutputToIndexDimension({0, 1, 2}, 0, {1, 2, 3}));

    return selectedFactors;
}

void ctcBeamSearchInputGradients(Matrix& inputDeltas, const Matrix& outputActivationWeights,
    const Matrix& inputPaths, const Matrix& outputDeltas, size_t beamSize)
{
    // {beamSize, miniBatchSize, timesteps}
    auto transitionPositions = applyToAdjacentElements(inputPaths, 2, NotEqual(), 0.0);

    // {beamSize, miniBatchSize, timesteps}
    auto scanPositions = inclusiveScan(transitionPositions, 2, Add(), 0.0);

    auto factors = computeFactors(scanPositions);

    // selected output deltas size is {beamSize, miniBatchSize, timesteps}
    // output deltas {alphabet, beamSize, miniBatchSize, timesteps}
    // scan positions {beamSize, miniBatchSize, timesteps}
    Matrix selectedOutputDeltas(scanPositions.size(), scanPositions.precision());

    indirectGather(selectedOutputDeltas, outputDeltas, scanPositions,
        MapOutputToIndexDimension({0, 1, 2}, 0, {1, 2, 3}));

    auto scaledOutputDeltas = apply(Matrix(factors), selectedOutputDeltas, Multiply());

    // scaled output deltas size is {beamSize, miniBatchSize, timesteps}
    // input paths is {beamSize, miniBatchSize, timesteps}
    // expanded input deltas {alphabet, beamSize, miniBatchSize, timesteps}
    Matrix expandedInputDeltas(outputDeltas.size(), outputDeltas.precision());

    indirectGather(expandedInputDeltas, scaledOutputDeltas, inputPaths,
        MapOutputToMatchingIndexDimension({1, 2, 3}, 0, {1, 2, 3}));

    reduce(inputDeltas, expandedInputDeltas, {1}, Add());
}

}
}


