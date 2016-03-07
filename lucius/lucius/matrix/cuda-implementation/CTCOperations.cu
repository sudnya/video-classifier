// Lucius Includes
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/CTCOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/Operation.h>
//CTC Includes
#include <lucius/matrix/ctc/interface/ctc.h>

// Standard Library Includes
#include <vector>
#include <thread>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }

namespace lucius
{
namespace matrix
{

void computeCtc(Matrix& costs, Matrix& gradients, const Matrix& inputActivations, const Matrix& reference)
{
    assert(costs.precision()            == SinglePrecision());
    assert(gradients.precision()        == SinglePrecision());
    assert(inputActivations.precision() == SinglePrecision());
    assert(reference.precision()        == SinglePrecision());

    //first call get_workspace_size to get workspace size
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

    auto labels = reduceGetPositions(reference, {0}, lucius::matrix::Maximum());
    
    for(auto element : labels)
    {
        labelsInMinibatch.push_back(static_cast<int>(element));
    }

    //call compute_ctc_loss
    compute_ctc_loss(static_cast<const float*>(inputActivations.data()), static_cast<float*>(gradients.data()), labelsInMinibatch.data(),
        labelLengthInMinibatch.data(), timeStepsInMinibatch.data(), vocabularySize, samplesPerMinibatch,
        static_cast<float*>(costs.data()), workspace.data(), runtimeInfo);

}
}
}

