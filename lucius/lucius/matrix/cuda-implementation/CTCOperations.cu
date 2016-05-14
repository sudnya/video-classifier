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

    assert(reference.size().size() >= 3);

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

    std::vector<int> timeStepsInMinibatch;
    std::vector<int> labelsInMinibatch;
    std::vector<int> labelLengthInMinibatch;

    // gather the labels
    // TODO: do this on the GPU, or use metadata
    for(size_t miniBatch = 0; miniBatch != samplesPerMinibatch; ++miniBatch)
    {
        bool labelEnded = false;

        for(size_t label = 0; label != labelSize; ++label)
        {
            double letterValue    = 0.0;
            size_t selectedLetter = 0;

            for(size_t letter = 0; letter != vocabularySize; ++letter)
            {
                double value = reference(letter, miniBatch, label);

                if(value > letterValue)
                {
                    letterValue    = value;
                    selectedLetter = letter;
                }
            }

            if(!labelEnded)
            {
                labelsInMinibatch.push_back(selectedLetter);
            }

            if(label == labelSize - 1)
            {
                if(!labelEnded)
                {
                    labelLengthInMinibatch.push_back(label + 1);
                }

                timeStepsInMinibatch.push_back(label + 1);
                break;
            }

            if((selectedLetter >= vocabularySize - 1) && (label > 0))
            {
                if(!labelEnded)
                {
                    labelLengthInMinibatch.push_back(label + 1);
                    labelEnded = true;
                }
                else
                {
                    timeStepsInMinibatch.push_back(label + 1);
                    break;
                }
            }
        }
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
        util::log("CTCOperations::Detail") << " reference " << reference.toString() << "\n";
        util::log("CTCOperations::Detail") << " costs " << costs.toString() << "\n";
        util::log("CTCOperations::Detail") << " input-gradients " << gradients.toString() << "\n";
    }

}

void computeCtc(Matrix& costs, Matrix& gradients,
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
            singleGradients = zeros(gradients.size(), singleReference.precision());
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

}
}

