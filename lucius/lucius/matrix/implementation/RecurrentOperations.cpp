
// Lucius Includes
#include <lucius/matrix/interface/RecurrentOperations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>

#include <lucius/matrix/interface/CudnnLibrary.h>
#include <lucius/matrix/interface/PrnnLibrary.h>
#include <lucius/matrix/interface/PrnnDescriptors.h>
#include <lucius/matrix/interface/CudnnDescriptors.h>

namespace lucius
{
namespace matrix
{

RecurrentOpsHandle::RecurrentOpsHandle(size_t layerSize, size_t miniBatchSize, size_t timesteps,
    size_t layers,
    RecurrentLayerDirection direction,
    RecurrentLayerType layerType,
    RecurrentLayerInputMode inputMode) :

    layerSize(layerSize),
    miniBatchSize(miniBatchSize),
    timesteps(timesteps),
    layers(layers),
    direction(direction),
    layerType(layerType),
    inputMode(inputMode)
{}

static std::string toString(RecurrentLayerDirection direction)
{
    switch(direction)
    {
    case RECURRENT_FORWARD:       return "forward";
    case RECURRENT_REVERSE:       return "reverse";
    case RECURRENT_BIDIRECTIONAL: return "bidirectional";
    }

    return "invalid";
}

static std::string toString(RecurrentLayerType type)
{
    switch(type)
    {
    case RECURRENT_SIMPLE_TYPE: return "relu";
    case RECURRENT_GRU_TYPE:    return "gru";
    case RECURRENT_LSTM_TYPE:   return "lstm";
    }

    return "invalid";
}

static std::string toString(RecurrentLayerInputMode mode)
{
    switch(mode)
    {
    case RECURRENT_LINEAR_INPUT: return "linear";
    case RECURRENT_SKIP_INPUT:   return "skip";
    }

    return "invalid";
}

std::string RecurrentOpsHandle::toString() const
{
    std::stringstream stream;

    stream <<
        "layerSize: " << layerSize << ", "
        "miniBatchSize: " << miniBatchSize << ", "
        "timesteps: " << timesteps << ", "
        "direction: " << matrix::toString(direction) << ", "
        "layerType: " << matrix::toString(layerType) << ", "
        "inputMode: " << matrix::toString(inputMode);

    return stream.str();
}

CudnnTensorDescriptor getCudnnInputDescriptor(const matrix::Matrix& input)
{
    return CudnnTensorDescriptor(input);
}

CudnnTensorDescriptor getCudnnInputDescriptor(const RecurrentOpsHandle& handle)
{
    return CudnnTensorDescriptor(
        Dimension(handle.layerSize, handle.miniBatchSize, handle.timesteps));
}

PrnnTensorDescriptor getPrnnInputDescriptor(const matrix::Matrix& input)
{
    return PrnnTensorDescriptor(input);
}

PrnnTensorDescriptor getPrnnInputDescriptor(const RecurrentOpsHandle& handle)
{
    return PrnnTensorDescriptor(
        Dimension(handle.layerSize, handle.miniBatchSize, handle.timesteps));
}

PrnnFilterDescriptor getPrnnFilterDescriptor(const matrix::Matrix& weights)
{
    return PrnnFilterDescriptor(weights);
}

PrnnFilterDescriptor getPrnnFilterDescriptor()
{
    return PrnnFilterDescriptor({});
}

CudnnFilterDescriptor getCudnnFilterDescriptor(const matrix::Matrix& weights)
{
    return CudnnFilterDescriptor(weights);
}

CudnnFilterDescriptor getCudnnFilterDescriptor()
{
    return CudnnFilterDescriptor({});
}

PrnnTensorDescriptor getPrnnSingleDescriptor(const RecurrentOpsHandle& handle)
{
    return PrnnTensorDescriptor(Dimension(1, handle.miniBatchSize, handle.timesteps));
}

CudnnTensorDescriptor getCudnnSingleDescriptor(const RecurrentOpsHandle& handle)
{
    return CudnnTensorDescriptor(Dimension(1, handle.miniBatchSize, handle.timesteps));
}

CudnnLibrary::cudnnDataType_t getCudnnDataType(const Precision& precision)
{
    if(precision == SinglePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_FLOAT;
    }
    else if(precision == DoublePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_DOUBLE;
    }
    else
    {
        return CudnnLibrary::CUDNN_DATA_HALF;
    }
}

matrix::Matrix createReserveRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    size_t reserveSize = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);
        PrnnTensorDescriptor inputDescriptor = getPrnnInputDescriptor(handle);

        PrnnLibrary::prnnGetRNNTrainingReserveSize(descriptor.descriptor(),
                                                   handle.timesteps,
                                                   &inputDescriptor.descriptor(),
                                                   &reserveSize);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle);
        CudnnTensorDescriptor inputDescriptor = getCudnnInputDescriptor(handle);

        CudnnLibrary::cudnnGetRNNTrainingReserveSize(descriptor.descriptor(),
                                                     handle.timesteps,
                                                     &inputDescriptor.descriptor(),
                                                     &reserveSize);
    }
    else
    {
        throw std::runtime_error("Tried to create recurrent ops reserve "
            "without CUDNN or Prnn libraries.");
    }

    return matrix::Matrix({reserveSize / precision.size()}, precision);
}

matrix::Matrix createWeightsRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    size_t weightsSize = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);
        PrnnTensorDescriptor inputDescriptor = getPrnnInputDescriptor(handle);

        PrnnLibrary::prnnGetRNNParamsSize(descriptor.descriptor(),
                                          inputDescriptor.descriptor(),
                                          &weightsSize);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle);
        CudnnTensorDescriptor inputDescriptor = getCudnnInputDescriptor(handle);

        CudnnLibrary::cudnnGetRNNParamsSize(descriptor.descriptor(),
                                            inputDescriptor.descriptor(),
                                            &weightsSize,
                                            getCudnnDataType(precision));
    }
    else
    {
        throw std::runtime_error("Tried to create recurrent ops weights "
            "without CUDNN or Prnn libraries.");
    }

    return matrix::Matrix({weightsSize / precision.size()}, precision);
}

matrix::Matrix sliceLayerWeights(const matrix::Matrix& weights, const RecurrentOpsHandle& handle,
    size_t layer, size_t offsetInLayer)
{
    size_t size   = 0;
    size_t offset = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);
        PrnnTensorDescriptor inputDescriptor = getPrnnInputDescriptor(handle);
        auto weightDescriptor = getPrnnFilterDescriptor(weights);
        PrnnFilterDescriptor filterDescriptor = getPrnnFilterDescriptor();

        PrnnLibrary::prnnGetRNNLinLayerMatrixParams(descriptor.descriptor(),
                                                    layer,
                                                    inputDescriptor.descriptor(),
                                                    weightDescriptor.descriptor(),
                                                    nullptr,
                                                    offsetInLayer,
                                                    filterDescriptor.descriptor(),
                                                    reinterpret_cast<void**>(&offset)
                                                    );

        size = filterDescriptor.dimensions().product();
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle);
        CudnnTensorDescriptor inputDescriptor = getCudnnInputDescriptor(handle);
        auto weightDescriptor = getPrnnFilterDescriptor(weights);
        CudnnFilterDescriptor filterDescriptor = getCudnnFilterDescriptor();

        CudnnLibrary::cudnnGetRNNLinLayerMatrixParams(descriptor.descriptor(),
                                                      layer,
                                                      inputDescriptor.descriptor(),
                                                      weightDescriptor.descriptor(),
                                                      nullptr,
                                                      offsetInLayer,
                                                      filterDescriptor.descriptor(),
                                                      reinterpret_cast<void**>(&offset)
                                                      );

        size = filterDescriptor.dimensions().product();
    }
    else
    {
        throw std::runtime_error("Tried to create slice out layer weights "
            "without CUDNN or Prnn libraries.");
    }

    return slice(weights, {offset}, {offset + size});
}

size_t getCudnnMatricesPerLayer(const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE)
    {
        return 4;
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        return 12;
    }
    else
    {
        return 16;
    }
}

size_t getMatricesPerLayer(const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE)
    {
        return 4;
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        return 12;
    }
    else
    {
        return 16;
    }
}

size_t getTotalWeightMatrices(const RecurrentOpsHandle& handle)
{
    return handle.layers * getMatricesPerLayer(handle);
}

static matrix::Matrix getPrnnWorkspace(const RecurrentOpsHandle& handle)
{
    size_t size = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);

        auto inputDescriptor = getPrnnInputDescriptor(handle);

        PrnnLibrary::prnnGetRNNWorkspaceSize(descriptor.descriptor(),
                                             handle.timesteps,
                                             &inputDescriptor.descriptor(),
                                             &size);
    }
    else
    {
        CudnnRNNDescriptor descriptor(handle);

        auto inputDescriptor = getCudnnInputDescriptor(handle);

        CudnnLibrary::cudnnGetRNNWorkspaceSize(descriptor.descriptor(),
                                               handle.timesteps,
                                               &inputDescriptor.descriptor(),
                                               &size);

    }
    else
    {
        throw std::runtime_error("Tried to call forwardPropRecurrent "
            "without CUDNN or Prnn libraries.");
    }
}

/** \brief Forward propagate through a recurrent weight matrix.
 *  \param weights The recurrent weight matrix.
 *   \param reserve Memory allocated for storing data needed for back propagation
 *                  (storage format determined by implementation)
 *  \param activations The input/output activations from the previous layer
 *                     (stored as [previous-layer-outputs, mini-batch, timesteps]).
 */
void forwardPropRecurrent(matrix::Matrix& outputActivations,
                          const matrix::Matrix& inputActivations,
                          matrix::Matrix& reserve,
                          const matrix::Matrix& weights,
                          const RecurrentOpsHandle& handle)
{
    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);

        auto xDescriptor = getPrnnInputDescriptor(inputActivations);
        auto yDescriptor = getPrnnInputDescriptor(outputActivations);
        auto workspace   = getPrnnWorkspace(handle);

        auto hxDescriptor = getPrnnSingleDescriptor(handle);
        auto cxDescriptor = getPrnnSingleDescriptor(handle);
        auto hyDescriptor = getPrnnSingleDescriptor(handle);
        auto cyDescriptor = getPrnnSingleDescriptor(handle);

        PrnnLibrary::prnnRNNForward(descriptor.descriptor(),
                                    handle.timesteps,
                                    xDescriptor.descriptor(),
                                    xDescriptor.data(),
                                    hxDescriptor.descriptor(),
                                    nullptr,
                                    cxDescriptor.descriptor(),
                                    nullptr,
                                    weightsDescriptor.descriptor(),
                                    nullptr,
                                    yDescriptor.descriptor(),
                                    yDescriptor.data(),
                                    hyDescriptor.descriptor(),
                                    nullptr,
                                    cyDescriptor.descriptor(),
                                    nullptr,
                                    workspace.data(),
                                    workspace.elements() * workspace.precision().size(),
                                    reserve.data(),
                                    reserve.elements() * reserve.precision().size());

    }
    else if(CudnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);

        auto xDescriptor = getCudnnInputDescriptor(inputActivations);
        auto yDescriptor = getCudnnInputDescriptor(outputActivations);
        auto workspace   = getCudnnWorkspace(handle);

        auto hxDescriptor = getCudnnSingleDescriptor(handle);
        auto cxDescriptor = getCudnnSingleDescriptor(handle);
        auto hyDescriptor = getCudnnSingleDescriptor(handle);
        auto cyDescriptor = getCudnnSingleDescriptor(handle);

        PrnnLibrary::prnnRNNForward(descriptor.descriptor(),
                                    handle.timesteps,
                                    xDescriptor.descriptor(),
                                    xDescriptor.data(),
                                    hxDescriptor.descriptor(),
                                    nullptr,
                                    cxDescriptor.descriptor(),
                                    nullptr,
                                    weightsDescriptor.descriptor(),
                                    nullptr,
                                    yDescriptor.descriptor(),
                                    yDescriptor.data(),
                                    hyDescriptor.descriptor(),
                                    nullptr,
                                    cyDescriptor.descriptor(),
                                    nullptr,
                                    workspace.data(),
                                    workspace.elements() * workspace.precision().size(),
                                    reserve.data(),
                                    reserve.elements() * reserve.precision().size());

    }
    else
    {
        throw std::runtime_error("Tried to call forwardPropRecurrent "
            "without CUDNN or Prnn libraries.");
    }

}

/** \brief Back propagate through a recurrent weight matrix, generating deltas.
 *  \param weights The recurrent weight matrix.
 *  \param deltas The input/output deltas from the previous layer
 *                (stored as [previous-layer-outputs, mini-batch, timesteps]).
 *   \param reserve Memory allocated for storing data needed for back propagation
 *                  (storage format determined by implementation)
 */
void backPropDeltasRecurrent(matrix::Matrix& inputDeltas,
    const matrix::Matrix& outputDeltas,
    const matrix::Matrix& weights,
    const matrix::Matrix& outputActivations,
    matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle)
{
    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);

        auto yDescriptor  = getPrnnInputDescriptor(outputActivations);
        auto dyDescriptor = getPrnnInputDescriptor(outputDeltas);
        auto dxDescriptor = getPrnnInputDescriptor(inputDeltas);
        auto workspace    = getPrnnWorkspace(handle);
        auto weights      = getPrnnFilterDescriptor(weights);

        auto hxDescriptor  = getPrnnSingleDescriptor(handle);
        auto cxDescriptor  = getPrnnSingleDescriptor(handle);
        auto dhxDescriptor = getPrnnSingleDescriptor(handle);
        auto dcxDescriptor = getPrnnSingleDescriptor(handle);
        auto dhyDescriptor = getPrnnSingleDescriptor(handle);
        auto dcyDescriptor = getPrnnSingleDescriptor(handle);

        PrnnLibrary::prnnRNNBackwardData(descriptor.descriptor(),
                                         handle.timesteps,
                                         yDescriptor.descriptor(),
                                         yDescriptor.data(),
                                         dyDescriptor.descriptor(),
                                         dyDescriptor.data(),
                                         dhyDescriptor.descriptor(),
                                         nullptr,
                                         dcyDescriptor.descriptor(),
                                         nullptr,
                                         weightDescriptor.descriptor(),
                                         weightDescriptor.data(),
                                         hxDescriptor.descriptor(),
                                         hxDescriptor.data(),
                                         cxDescriptor.descriptor(),
                                         cxDescriptor.data(),
                                         &dxDescriptor.descriptor(),
                                         dxDescriptor.data(),
                                         dhxDescriptor.descriptor(),
                                         dhxDescriptor.data(),
                                         dcxDescriptor.descriptor(),
                                         dcxDescriptor.data(),
                                         workspace.data(),
                                         workspace.elements() * workspace.precision().size(),
                                         reserve.data(),
                                         reserve.elements() * reserve.precision().size());

    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle);

        auto yDescriptor  = getCudnnInputDescriptor(outputActivations);
        auto dyDescriptor = getCudnnInputDescriptor(outputDeltas);
        auto dxDescriptor = getCudnnInputDescriptor(inputDeltas);
        auto workspace    = getCudnnWorkspace(handle);
        auto weights      = getCudnnFilterDescriptor(weights);

        auto hxDescriptor  = getCudnnSingleDescriptor(handle);
        auto cxDescriptor  = getCudnnSingleDescriptor(handle);
        auto dhxDescriptor = getCudnnSingleDescriptor(handle);
        auto dcxDescriptor = getCudnnSingleDescriptor(handle);
        auto dhyDescriptor = getCudnnSingleDescriptor(handle);
        auto dcyDescriptor = getCudnnSingleDescriptor(handle);

        PrnnLibrary::cudnnRNNBackwardData(descriptor.descriptor(),
                                          handle.timesteps,
                                          yDescriptor.descriptor(),
                                          yDescriptor.data(),
                                          dyDescriptor.descriptor(),
                                          dyDescriptor.data(),
                                          dhyDescriptor.descriptor(),
                                          nullptr,
                                          dcyDescriptor.descriptor(),
                                          nullptr,
                                          weightDescriptor.descriptor(),
                                          weightDescriptor.data(),
                                          hxDescriptor.descriptor(),
                                          hxDescriptor.data(),
                                          cxDescriptor.descriptor(),
                                          cxDescriptor.data(),
                                          &dxDescriptor.descriptor(),
                                          dxDescriptor.data(),
                                          dhxDescriptor.descriptor(),
                                          dhxDescriptor.data(),
                                          dcxDescriptor.descriptor(),
                                          dcxDescriptor.data(),
                                          workspace.data(),
                                          workspace.elements() * workspace.precision().size(),
                                          reserve.data(),
                                          reserve.elements() * reserve.precision().size());

    }
    else
    {
        throw std::runtime_error("Tried to call backPropDeltasRecurrent "
            "without CUDNN or Prnn libraries.");
    }

}

/** \brief Compute gradient for the recurrent weight matrix.
 *  \param deltas Deltas for the layer.
 *  \param dWeights The output gradients.
 *   \param reserve Memory allocated for storing data needed for back propagation
 *                  (storage format determined by implementation)
 */
void backPropGradientsRecurrent(matrix::Matrix& dWeights,
    const matrix::Matrix& inputActivations,
    const matrix::Matrix& outputActivations,
    const matrix::Matrix& deltas,
    const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle)
{
    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle);

        auto yDescriptor = getPrnnInputDescriptor(outputActivations);
        auto xDescriptor = getPrnnInputDescriptor(inputActivations);
        auto workspace   = getPrnnWorkspace(handle);
        auto dWeights    = getPrnnFilterDescriptor(dWeights);

        auto hxDescriptor = getPrnnSingleDescriptor(handle);

        PrnnLibrary::prnnRNNBackwardWeights(descriptor.descriptor(),
                                            handle.timesteps,
                                            xDescriptor.descriptor(),
                                            xDescriptor.data(),
                                            hxDescriptor.descriptor(),
                                            hxDescriptor.data(),
                                            yDescriptor.descriptor(),
                                            yDescriptor.data(),
                                            workspace.data(),
                                            workspace.elements() * workspace.precision().size(),
                                            dWeights.descriptor(),
                                            dWeights.data(),
                                            reserve.data(),
                                            reserve.elements() * reserve.precision().size());

    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle);

        auto yDescriptor = getCudnnInputDescriptor(outputActivations);
        auto xDescriptor = getCudnnInputDescriptor(inputActivations);
        auto workspace   = getCudnnWorkspace(handle);
        auto dWeights    = getCudnnFilterDescriptor(dWeights);

        auto hxDescriptor = getCudnnSingleDescriptor(handle);

        PrnnLibrary::cudnnRNNBackwardWeights(descriptor.descriptor(),
                                             handle.timesteps,
                                             xDescriptor.descriptor(),
                                             xDescriptor.data(),
                                             hxDescriptor.descriptor(),
                                             hxDescriptor.data(),
                                             yDescriptor.descriptor(),
                                             yDescriptor.data(),
                                             workspace.data(),
                                             workspace.elements() * workspace.precision().size(),
                                             dWeights.descriptor(),
                                             dWeights.data(),
                                             reserve.data(),
                                             reserve.elements() * reserve.precision().size());

    }
    else
    {
        throw std::runtime_error("Tried to call backPropDeltasRecurrent "
            "without CUDNN or PRNN libraries.");
    }

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



