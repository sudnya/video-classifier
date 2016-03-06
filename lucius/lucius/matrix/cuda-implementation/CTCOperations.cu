// Lucius Includes
#include <lucius/matrix/interface/CTCOperations.h>

// Standard Library Includes
#include <thread>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }

namespace lucius
{
namespace matrix
{
struct ctcComputeInfo {
    ctcComputeLocation loc;
    union {
        unsigned int num_threads;
        CUstream stream;
    };
};

void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations, const Matrix& reference)
{
    assert(costs.precision() == matrix::SinglePrecision());
    assert(gradients.precision() == matrix::SinglePrecision());
    assert(inputActivations.precision() == matrix::SinglePrecision());
    assert(reference.precision() == matrix::SinglePrecision());

    //first call get_workspace_size to get workspace size
    size_t sizeBytes = 0;
    ctcComputeInfo runtimeInfo;
    
    if(parallel::isCudaEnabled())
    {
        runtimeInfo.loc = CTC_GPU;
        runtimeInfo.stream = 0;
    }
    else
    {
        runtimeInfo.loc = CTC_CPU;
        runtimeInfo.num_threads = std::thread::hardware_concurrency();
    }

    //if () on CPU => set num_threads
    //else set stream

    size_t vocabularySize      = reference.size().front();
    size_t samplesPerMinibatch = reference.size()[reference.size().size() - 2];

    std::vector<int> labelLengthInMinibatch;
    std::vector<int> timeStepsInMinibatch;

    for(size_t sample = 0; sample < samplesPerMinibatch; ++sample)
    {
        labelLengthInMinibatch.push_back(reference.size()[reference.size().size() - 1]);
        timeStepsInMinibatch.push_back(inputActivations.size()[inputActivations.size().size() - 1]);
    }

    get_workspace_size(labelLengthInMinibatch.data(), timeStepsInMinibatch.data(), vocabularySize, samplesPerMinibatch, runtimeInfo, &sizeBytes);

    Allocation workspace(sizeBytes);

    std::vector<int> labelsInMinibatch;

    auto labels = reduceGetPositions(reference, {0}, matrix::Maximum());
    
    for(auto& element : labels)
    {
        labelsInMinibatch.push_back(static_cast<int>(element));
    }

    //call compute_ctc_loss
    compute_ctc_loss(inputActivations.data(), gradients.data(), labelsInMinibatch.data(),
        labelLengthInMinibatch.data(), timeStepsInMinibatch.data(), vocabularySize, samplesPerMinibatch,
        costs.data(), workspace.data(), runtimeInfo);

}
}
}

