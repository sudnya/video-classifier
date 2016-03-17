// Lucius Includes
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/CTCOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/debug.h>

//CTC Includes
#include <lucius/matrix/ctc/interface/ctc.h>

// Standard Library Includes
#include <vector>
#include <thread>

// Forward Declarations
namespace lucius
{
namespace matrix
{

static void computeCtcOnSinglePrecisionSequence(Matrix& costs, Matrix& gradients,
    const Matrix& inputActivations, const Matrix& reference)
{
    assert(costs.precision()            == SinglePrecision());
    assert(gradients.precision()        == SinglePrecision());
    assert(inputActivations.precision() == SinglePrecision());
    assert(reference.precision()        == SinglePrecision());

    size_t sizeBytes = 0;
    lucius::matrix::ctc::ctcComputeInfo runtimeInfo;

    if(parallel::isCudaEnabled())
    {
        runtimeInfo.loc = lucius::matrix::ctc::ctcComputeLocation::CTC_GPU;
        runtimeInfo.stream = 0;
    }
    else
    {
        runtimeInfo.loc = lucius::matrix::ctc::ctcComputeLocation::CTC_CPU;
        runtimeInfo.num_threads = std::thread::hardware_concurrency();
    }

    size_t vocabularySize      = reference.size().front();
    size_t samplesPerMinibatch = reference.size()[reference.size().size() - 2];
    size_t labelSize           = reference.size()[reference.size().size() - 1];
    size_t timesteps           = inputActivations.size()[inputActivations.size().size() - 1];

    std::vector<int> labelLengthInMinibatch;
    std::vector<int> timeStepsInMinibatch;

    for(size_t sample = 0; sample < samplesPerMinibatch; ++sample)
    {
        labelLengthInMinibatch.push_back(labelSize);
        timeStepsInMinibatch.push_back(timesteps);
    }

    get_workspace_size(labelLengthInMinibatch.data(), timeStepsInMinibatch.data(),
        vocabularySize, samplesPerMinibatch, runtimeInfo, &sizeBytes);

    Allocation workspace(sizeBytes);

    std::vector<int> labelsInMinibatch;

    // TODO: do this on the GPU
    for(size_t miniBatch = 0; miniBatch != samplesPerMinibatch; ++miniBatch)
    {
        for(size_t timestep = 0; timestep != timesteps; ++timestep)
        {
            double labelValue = 0.0;
            size_t label      = 0;

            for(size_t letter = 0; letter != vocabularySize; ++letter)
            {
                double value = reference(label, miniBatch, timestep);

                if(value > labelValue)
                {
                    labelValue = value;
                    label      = letter;
                }
            }

            labelsInMinibatch.push_back(label);
        }
    }

    //call compute_ctc_loss
    compute_ctc_loss(static_cast<const float*>(inputActivations.data()),
        static_cast<float*>(gradients.data()), labelsInMinibatch.data(),
        labelLengthInMinibatch.data(), timeStepsInMinibatch.data(), vocabularySize,
        samplesPerMinibatch, static_cast<float*>(costs.data()), workspace.data(), runtimeInfo);

}

static void computeCtcOnSingleSequence(Matrix& costs, Matrix& gradients,
    const Matrix& inputActivations, const Matrix& reference)
{
    if(costs.precision() == SinglePrecision())
    {
        computeCtcOnSinglePrecisionSequence(costs, gradients, inputActivations, reference);
    }
    else
    {
        auto singleInputActivations = copy(inputActivations, SinglePrecision());
        auto singleReference = copy(reference, SinglePrecision());

        Matrix singleGradients;

        if(!gradients.empty())
        {
            singleGradients = Matrix(gradients.size(), singleReference.precision());
        }

        Matrix singleCosts(costs.size(), singleReference.precision());

        computeCtcOnSinglePrecisionSequence(singleCosts, singleGradients, singleInputActivations,
            singleReference);

        copy(costs, singleCosts);

        if(!gradients.empty())
        {
            copy(gradients, singleGradients);
        }
    }
}

static void runCtcOnSlice(Matrix& costs, Matrix& gradients,
    const Matrix& inputActivations, const Matrix& reference, size_t startTimestep,
    size_t timestep, size_t sample)
{
    auto start = zeros(reference.size());
    auto end   = reference.size();

    start[start.size() - 2] = sample;
    end  [end.size()   - 2] = sample + 1;

    start[start.size() - 1] = startTimestep;
    end  [end.size()   - 1] = timestep;

    Matrix gradientsSlice;

    auto costsSlice = slice(costs, {sample}, {sample + 1});

    if(!gradients.empty())
    {
        gradientsSlice  = slice(gradients, start, end);
    }

    auto inputActivationsSlice = copy(slice(inputActivations, start, end));
    auto referenceSlice        = copy(slice(reference,        start, end));

    auto costsCopy  = zeros({1}, costs.precision());

    Matrix gradientsCopy;

    if (!gradients.empty())
    {
        gradientsCopy = zeros(gradientsSlice.size(), gradientsSlice.precision());
    }

    computeCtcOnSingleSequence(costsCopy, gradientsCopy, inputActivationsSlice,
        referenceSlice);

    copy(costsSlice, costsCopy);

    if(!gradients.empty())
    {
        copy(gradientsSlice, gradientsCopy);
    }
}

void computeCtc(Matrix& costs, Matrix& gradients,
    const Matrix& inputActivations, const Matrix& reference)
{
    if(true)
    {
        computeCtcOnSingleSequence(costs, gradients, inputActivations, reference);
        return;
    }

    size_t miniBatchSize = reference.size()[reference.size().size() - 2];
    size_t timesteps     = reference.size()[reference.size().size() - 1];
    size_t layerSize     = reference.size().product() / (miniBatchSize * timesteps);

    for(size_t sample = 0; sample < miniBatchSize; ++sample)
    {
        size_t startTimestep = 0;

        for(size_t timestep = 0; timestep < timesteps; ++timestep)
        {
            if(reference(layerSize - 1, sample, timestep) == 0.0)
            {
                continue;
            }

            runCtcOnSlice(costs, gradients, inputActivations, reference,
                startTimestep, timesteps, sample);

            startTimestep = timestep + 1;
            timestep      = timestep + 1;
        }

        if(startTimestep < timesteps)
        {
            runCtcOnSlice(costs, gradients, inputActivations, reference,
                startTimestep, timesteps, sample);
        }
    }
}

}
}

