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
    const Matrix& inputActivations, const LabelVector& labels,
    const IndexVector& timestepsPerSample)
{
    assert(costs.precision()            == SinglePrecision());
    assert(gradients.precision()        == SinglePrecision());
    assert(inputActivations.precision() == SinglePrecision());

    size_t sizeBytes = 0;
    lucius::matrix::ctc::ctcComputeInfo runtimeInfo;

    if(parallel::isCudaEnabled())
    {
        parallel::setNotSynchronized();

        runtimeInfo.loc = lucius::matrix::ctc::ctcComputeLocation::CTC_GPU;
        runtimeInfo.stream = 0;
    }
    else
    {
        runtimeInfo.loc = lucius::matrix::ctc::ctcComputeLocation::CTC_CPU;
        runtimeInfo.num_threads = std::thread::hardware_concurrency();
    }

    size_t vocabularySize      = inputActivations.size().front();
    size_t samplesPerMinibatch = inputActivations.size()[inputActivations.size().size() - 2];

    std::vector<int> timeStepsInMinibatch;
    std::vector<int> labelsInMinibatch;
    std::vector<int> labelLengthInMinibatch;

    for(size_t miniBatch = 0; miniBatch != samplesPerMinibatch; ++miniBatch)
    {
        timeStepsInMinibatch.push_back(timestepsPerSample[miniBatch]);

        for(auto& label : labels[miniBatch])
        {
            labelsInMinibatch.push_back(label);
        }

        labelLengthInMinibatch.push_back(labels[miniBatch].size());
    }

    get_workspace_size(labelLengthInMinibatch.data(), timeStepsInMinibatch.data(),
        vocabularySize, samplesPerMinibatch, runtimeInfo, &sizeBytes);

    Allocation workspace(sizeBytes);

    //call compute_ctc_loss
    auto status = compute_ctc_loss(static_cast<const float*>(inputActivations.data()),
        static_cast<float*>(gradients.data()), labelsInMinibatch.data(),
        labelLengthInMinibatch.data(), timeStepsInMinibatch.data(), vocabularySize,
        samplesPerMinibatch, static_cast<float*>(costs.data()), workspace.data(), runtimeInfo);

    if (status != ctc::CTC_STATUS_SUCCESS)
    {
        throw std::runtime_error("CTC operation failed.");
    }

    if(util::isLogEnabled("CTCOperations::Detail"))
    {
        util::log("CTCOperations::Detail") << " activations "
            << inputActivations.toString() << "\n";
        util::log("CTCOperations::Detail") << " costs " << costs.toString() << "\n";
        util::log("CTCOperations::Detail") << " input-gradients " << gradients.toString() << "\n";
    }

}

void computeCtc(Matrix& costs, Matrix& gradients,
    const Matrix& inputActivations, const LabelVector& labels,
    const IndexVector& timestepsPerSample)
{
    if(costs.precision() == SinglePrecision())
    {
        computeCtcOnSinglePrecisionSequence(costs, gradients, inputActivations,
            labels, timestepsPerSample);
    }
    else
    {
        auto singleInputActivations = copy(inputActivations, SinglePrecision());

        Matrix singleGradients;

        if(!gradients.empty())
        {
            singleGradients = zeros(gradients.size(), SinglePrecision());
        }

        Matrix singleCosts(costs.size(), singleGradients.precision());

        computeCtcOnSinglePrecisionSequence(singleCosts, singleGradients, singleInputActivations,
            labels, timestepsPerSample);

        copy(costs, singleCosts);

        if(!gradients.empty())
        {
            copy(gradients, singleGradients);
        }
    }
}

}
}

