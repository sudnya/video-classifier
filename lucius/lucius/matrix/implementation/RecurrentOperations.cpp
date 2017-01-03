
// Lucius Includes
#include <lucius/matrix/interface/RecurrentOperations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/NativeRecurrentOperations.h>

#include <lucius/matrix/interface/CudnnLibrary.h>
#include <lucius/matrix/interface/PrnnLibrary.h>
#include <lucius/matrix/interface/PrnnDescriptors.h>
#include <lucius/matrix/interface/CudnnDescriptors.h>

#include <lucius/parallel/interface/Synchronization.h>

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
    case RECURRENT_SIMPLE_TYPE:      return "relu";
    case RECURRENT_SIMPLE_TANH_TYPE: return "tanh";
    case RECURRENT_GRU_TYPE:         return "gru";
    case RECURRENT_LSTM_TYPE:        return "lstm";
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

CudnnTensorDescriptorArray getCudnnTensorDescriptorArray(const matrix::Matrix& activations)
{
    matrix::Dimension dimensions = {1,   activations.size()[0], activations.size()[1]  };
    matrix::Dimension strides    = {1, activations.stride()[0], activations.stride()[1]};

    size_t timesteps = activations.size()[2];

    return CudnnTensorDescriptorArray(const_cast<void*>(activations.data()), dimensions,
        strides, timesteps, activations.precision());
}

CudnnTensorDescriptorArray getCudnnTensorDescriptorArray(
    const RecurrentOpsHandle& handle, const Precision& precision)
{
    return CudnnTensorDescriptorArray({1, handle.layerSize, handle.miniBatchSize},
                                      {1, 1, handle.layerSize},
                                      handle.timesteps,
                                      precision);
}

PrnnTensorDescriptor getPrnnTensorDescriptor()
{
    return PrnnTensorDescriptor(Matrix({1, 1, 1}));
}

PrnnTensorDescriptor getPrnnTensorDescriptor(const matrix::Matrix& activations)
{
    return PrnnTensorDescriptor(activations);
}

PrnnTensorDescriptorArray getPrnnTensorDescriptorArray(const matrix::Matrix& activations)
{
    matrix::Dimension dimensions = {1,   activations.size()[0], activations.size()[1]  };
    matrix::Dimension strides    = {1, activations.stride()[0], activations.stride()[1]};

    size_t timesteps = activations.size()[2];

    return PrnnTensorDescriptorArray(const_cast<void*>(activations.data()), dimensions,
        strides, timesteps, activations.precision());
}

PrnnTensorDescriptorArray getPrnnTensorDescriptorArray(
    const RecurrentOpsHandle& handle, const Precision& precision)
{
    return PrnnTensorDescriptorArray({1, handle.layerSize, handle.miniBatchSize},
                                     {1, 1, handle.layerSize},
                                     handle.timesteps,
                                     precision);
}

CudnnFilterDescriptor getCudnnFilterDescriptor(const matrix::Matrix& weights)
{
    return CudnnFilterDescriptor(reshape(weights, {1, 1, weights.size().front()}));
}

CudnnFilterDescriptor getCudnnFilterDescriptor()
{
    return CudnnFilterDescriptor(Matrix({1, 1, 1}));
}

PrnnTensorDescriptor getPrnnSingleDescriptor(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    auto size = Dimension(handle.layerSize, handle.miniBatchSize, handle.timesteps);

    return PrnnTensorDescriptor(size, linearStride(size), precision);
}

CudnnTensorDescriptor getCudnnSingleDescriptor(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    auto size = Dimension(1, handle.miniBatchSize, handle.timesteps);

    return CudnnTensorDescriptor(size, linearStride(size), precision);
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
        PrnnRNNDescriptor descriptor(handle, precision);
        auto inputDescriptors = getPrnnTensorDescriptorArray(handle, precision);

        PrnnLibrary::prnnGetRNNTrainingReserveSize(descriptor.descriptor(),
                                                   handle.timesteps,
                                                   inputDescriptors.descriptors(),
                                                   &reserveSize);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle, precision);
        auto inputDescriptors = getCudnnTensorDescriptorArray(handle, precision);

        CudnnLibrary::cudnnGetRNNTrainingReserveSize(descriptor.descriptor(),
                                                     handle.timesteps,
                                                     inputDescriptors.descriptors(),
                                                     &reserveSize);
    }
    else
    {
        reserveSize = nativeRNNGetTrainingReserveSize(handle, precision);
    }

    return matrix::Matrix({reserveSize / precision.size()}, precision);
}

matrix::Matrix createWeightsRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision)
{
    size_t weightsSize = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle, precision);
        auto inputDescriptor = getPrnnTensorDescriptorArray(handle, precision);

        PrnnLibrary::prnnGetRNNParamsSize(descriptor.descriptor(),
                                          *inputDescriptor.descriptors(),
                                          &weightsSize);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle, precision);
        auto inputDescriptor = getCudnnTensorDescriptorArray(handle, precision);

        CudnnLibrary::cudnnGetRNNParamsSize(descriptor.descriptor(),
                                            *inputDescriptor.descriptors(),
                                            &weightsSize,
                                            getCudnnDataType(precision));
    }
    else
    {
        weightsSize = nativeRNNGetWeightsSize(handle, precision);
    }

    return matrix::Matrix({weightsSize / precision.size()}, precision);
}

static matrix::Matrix sliceLayerLinearMatrix(const matrix::Matrix& weights,
    const RecurrentOpsHandle& handle, size_t layer, size_t offsetInLayer)
{
    size_t size   = 0;
    size_t offset = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle, weights.precision());
        auto inputDescriptor = getPrnnTensorDescriptorArray(handle, weights.precision());
        auto weightDescriptor = getPrnnTensorDescriptor(weights);
        PrnnTensorDescriptor filterDescriptor = getPrnnTensorDescriptor();

        PrnnLibrary::prnnGetRNNLinLayerMatrixParams(descriptor.descriptor(),
                                                    layer,
                                                    *inputDescriptor.descriptors(),
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
        CudnnRNNDescriptor descriptor(handle, weights.precision());
        auto inputDescriptor = getCudnnTensorDescriptorArray(handle, weights.precision());
        auto weightDescriptor = getCudnnFilterDescriptor(weights);
        CudnnFilterDescriptor filterDescriptor = getCudnnFilterDescriptor();

        CudnnLibrary::cudnnGetRNNLinLayerMatrixParams(descriptor.descriptor(),
                                                      layer,
                                                      *inputDescriptor.descriptors(),
                                                      weightDescriptor.descriptor(),
                                                      nullptr,
                                                      offsetInLayer,
                                                      filterDescriptor.descriptor(),
                                                      reinterpret_cast<void**>(&offset)
                                                      );

        size = filterDescriptor.getDimensions().product();
    }
    else
    {
        std::tie(size, offset) = nativeRNNGetLinearLayerMatrixSizeAndOffset(
            handle, weights.precision(), layer, offsetInLayer);

        size /= weights.precision().size();
    }

    offset /= weights.precision().size();

    return slice(weights, {offset}, {offset + size});
}

static matrix::Matrix sliceLayerBiasMatrix(const matrix::Matrix& weights,
    const RecurrentOpsHandle& handle, size_t layer, size_t offsetInLayer)
{
    size_t size   = 0;
    size_t offset = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle, weights.precision());
        auto inputDescriptor = getPrnnTensorDescriptorArray(handle, weights.precision());
        auto weightDescriptor = getPrnnTensorDescriptor(weights);
        PrnnTensorDescriptor filterDescriptor = getPrnnTensorDescriptor();

        PrnnLibrary::prnnGetRNNLinLayerBiasParams(descriptor.descriptor(),
                                                  layer,
                                                  *inputDescriptor.descriptors(),
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
        CudnnRNNDescriptor descriptor(handle, weights.precision());
        auto inputDescriptor = getCudnnTensorDescriptorArray(handle, weights.precision());
        auto weightDescriptor = getCudnnFilterDescriptor(weights);
        CudnnFilterDescriptor filterDescriptor = getCudnnFilterDescriptor();

        CudnnLibrary::cudnnGetRNNLinLayerBiasParams(descriptor.descriptor(),
                                                    layer,
                                                    *inputDescriptor.descriptors(),
                                                    weightDescriptor.descriptor(),
                                                    nullptr,
                                                    offsetInLayer,
                                                    filterDescriptor.descriptor(),
                                                    reinterpret_cast<void**>(&offset)
                                                    );

        size = filterDescriptor.getDimensions().product();
    }
    else
    {
        std::tie(size, offset) = nativeRNNGetBiasLayerMatrixSizeAndOffset(
            handle, weights.precision(), layer, offsetInLayer);

        size /= weights.precision().size();
    }

    offset /= weights.precision().size();

    return slice(weights, {offset}, {offset + size});
}

matrix::Matrix sliceLayerWeights(const matrix::Matrix& weights, const RecurrentOpsHandle& handle,
    size_t layer, size_t offsetInLayer)
{
    if(isBiasMatrix(handle, offsetInLayer))
    {
        return sliceLayerBiasMatrix(weights, handle, layer, offsetInLayer / 2);
    }
    else
    {
        return sliceLayerLinearMatrix(weights, handle, layer, offsetInLayer / 2);
    }
}

size_t getDirectionMultiplier(const RecurrentOpsHandle& handle)
{
    return handle.direction == RECURRENT_BIDIRECTIONAL ? 2 : 1;
}

size_t getMatricesPerLayer(const RecurrentOpsHandle& handle)
{
    if(handle.layerType == RECURRENT_SIMPLE_TYPE ||
        handle.layerType == RECURRENT_SIMPLE_TANH_TYPE)
    {
        return 4 * getDirectionMultiplier(handle);
    }
    else if(handle.layerType == RECURRENT_GRU_TYPE)
    {
        return 12 * getDirectionMultiplier(handle);
    }
    else
    {
        return 16 * getDirectionMultiplier(handle);
    }
}

matrix::Matrix sliceLayerWeights(const matrix::Matrix& weights, const RecurrentOpsHandle& handle,
    size_t offset)
{
    return sliceLayerWeights(weights, handle, offset / getMatricesPerLayer(handle),
        offset % getMatricesPerLayer(handle));
}

size_t getTotalWeightMatrices(const RecurrentOpsHandle& handle)
{
    return handle.layers * getMatricesPerLayer(handle);
}

bool isBiasMatrix(const RecurrentOpsHandle& handle, size_t index)
{
    return index % 2 == 1;
}

static matrix::Matrix getWorkspace(const RecurrentOpsHandle& handle, const Precision& precision)
{
    size_t size = 0;

    if(PrnnLibrary::loaded())
    {
        PrnnRNNDescriptor descriptor(handle, precision);

        auto inputDescriptors = getPrnnTensorDescriptorArray(handle, precision);

        PrnnLibrary::prnnGetRNNWorkspaceSize(descriptor.descriptor(),
                                             handle.timesteps,
                                             inputDescriptors.descriptors(),
                                             &size);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle, precision);

        auto inputDescriptors = getCudnnTensorDescriptorArray(handle, precision);

        CudnnLibrary::cudnnGetRNNWorkspaceSize(descriptor.descriptor(),
                                               handle.timesteps,
                                               inputDescriptors.descriptors(),
                                               &size);

    }
    else
    {
        return 0;
    }

    return size;
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
        parallel::setNotSynchronized();

        PrnnRNNDescriptor descriptor(handle, weights.precision());

        auto xDescriptor = getPrnnTensorDescriptorArray(inputActivations);
        auto yDescriptor = getPrnnTensorDescriptorArray(outputActivations);
        auto workspace   = getWorkspace(handle, weights.precision());

        auto weightsDescriptor = getPrnnTensorDescriptor(weights);

        auto hxDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto cxDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto hyDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto cyDescriptor = getPrnnSingleDescriptor(handle, weights.precision());

        PrnnLibrary::prnnRNNForward(descriptor.descriptor(),
                                    handle.timesteps,
                                    xDescriptor.descriptors(),
                                    xDescriptor.data(),
                                    hxDescriptor.descriptor(),
                                    nullptr,
                                    cxDescriptor.descriptor(),
                                    nullptr,
                                    weightsDescriptor.descriptor(),
                                    weightsDescriptor.data(),
                                    yDescriptor.descriptors(),
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
        parallel::setNotSynchronized();

        CudnnRNNDescriptor descriptor(handle, weights.precision());

        auto xDescriptor = getCudnnTensorDescriptorArray(inputActivations);
        auto yDescriptor = getCudnnTensorDescriptorArray(outputActivations);
        auto workspace   = getWorkspace(handle, weights.precision());

        auto weightsDescriptor = getCudnnFilterDescriptor(weights);

        auto hxDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto cxDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto hyDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto cyDescriptor = getCudnnSingleDescriptor(handle, weights.precision());

        CudnnLibrary::cudnnRNNForwardTraining(descriptor.descriptor(),
                                              handle.timesteps,
                                              xDescriptor.descriptors(),
                                              xDescriptor.data(),
                                              hxDescriptor.descriptor(),
                                              nullptr,
                                              cxDescriptor.descriptor(),
                                              nullptr,
                                              weightsDescriptor.descriptor(),
                                              weightsDescriptor.data(),
                                              yDescriptor.descriptors(),
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
        nativeRNNForwardPropRecurrent(outputActivations,
                                      inputActivations,
                                      reserve,
                                      weights,
                                      handle);
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
        parallel::setNotSynchronized();

        PrnnRNNDescriptor descriptor(handle, weights.precision());

        auto yDescriptor  = getPrnnTensorDescriptorArray(outputActivations);
        auto dyDescriptor = getPrnnTensorDescriptorArray(outputDeltas);
        auto dxDescriptor = getPrnnTensorDescriptorArray(inputDeltas);
        auto workspace    = getWorkspace(handle, weights.precision());

        auto weightsDescriptor = getPrnnTensorDescriptor(weights);

        auto hxDescriptor  = getPrnnSingleDescriptor(handle, weights.precision());
        auto cxDescriptor  = getPrnnSingleDescriptor(handle, weights.precision());
        auto dhxDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto dcxDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto dhyDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto dcyDescriptor = getPrnnSingleDescriptor(handle, weights.precision());

        PrnnLibrary::prnnRNNBackwardData(descriptor.descriptor(),
                                         handle.timesteps,
                                         yDescriptor.descriptors(),
                                         yDescriptor.data(),
                                         dyDescriptor.descriptors(),
                                         dyDescriptor.data(),
                                         dhyDescriptor.descriptor(),
                                         nullptr,
                                         dcyDescriptor.descriptor(),
                                         nullptr,
                                         weightsDescriptor.descriptor(),
                                         weightsDescriptor.data(),
                                         hxDescriptor.descriptor(),
                                         nullptr,
                                         cxDescriptor.descriptor(),
                                         nullptr,
                                         dxDescriptor.descriptors(),
                                         dxDescriptor.data(),
                                         dhxDescriptor.descriptor(),
                                         nullptr,
                                         dcxDescriptor.descriptor(),
                                         nullptr,
                                         workspace.data(),
                                         workspace.elements() * workspace.precision().size(),
                                         reserve.data(),
                                         reserve.elements() * reserve.precision().size());

    }
    else if(CudnnLibrary::loaded())
    {
        parallel::setNotSynchronized();

        CudnnRNNDescriptor descriptor(handle, weights.precision());

        auto yDescriptor  = getCudnnTensorDescriptorArray(outputActivations);
        auto dyDescriptor = getCudnnTensorDescriptorArray(outputDeltas);
        auto dxDescriptor = getCudnnTensorDescriptorArray(inputDeltas);
        auto workspace    = getWorkspace(handle, weights.precision());

        auto weightsDescriptor = getCudnnFilterDescriptor(weights);

        auto hxDescriptor  = getCudnnSingleDescriptor(handle, weights.precision());
        auto cxDescriptor  = getCudnnSingleDescriptor(handle, weights.precision());
        auto dhxDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto dcxDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto dhyDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto dcyDescriptor = getCudnnSingleDescriptor(handle, weights.precision());

        CudnnLibrary::cudnnRNNBackwardData(descriptor.descriptor(),
                                           handle.timesteps,
                                           yDescriptor.descriptors(),
                                           yDescriptor.data(),
                                           dyDescriptor.descriptors(),
                                           dyDescriptor.data(),
                                           dhyDescriptor.descriptor(),
                                           nullptr,
                                           dcyDescriptor.descriptor(),
                                           nullptr,
                                           weightsDescriptor.descriptor(),
                                           weightsDescriptor.data(),
                                           hxDescriptor.descriptor(),
                                           nullptr,
                                           cxDescriptor.descriptor(),
                                           nullptr,
                                           dxDescriptor.descriptors(),
                                           dxDescriptor.data(),
                                           dhxDescriptor.descriptor(),
                                           nullptr,
                                           dcxDescriptor.descriptor(),
                                           nullptr,
                                           workspace.data(),
                                           workspace.elements() * workspace.precision().size(),
                                           reserve.data(),
                                           reserve.elements() * reserve.precision().size());

    }
    else
    {
        nativeRNNBackPropDeltasRecurrent(inputDeltas,
                                         outputDeltas,
                                         weights,
                                         outputActivations,
                                         reserve,
                                         handle);
    }
}

/** \brief Compute gradient for the recurrent weight matrix.
 *  \param dWeights The output gradients.
 *   \param reserve Memory allocated for storing data needed for back propagation
 *                  (storage format determined by implementation)
 */
void backPropGradientsRecurrent(matrix::Matrix& dWeights,
    const matrix::Matrix& inputActivations,
    const matrix::Matrix& outputActivations,
    const matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle)
{
    if(PrnnLibrary::loaded())
    {
        parallel::setNotSynchronized();

        PrnnRNNDescriptor descriptor(handle, dWeights.precision());

        auto yDescriptor = getPrnnTensorDescriptorArray(outputActivations);
        auto xDescriptor = getPrnnTensorDescriptorArray(inputActivations);
        auto workspace   = getWorkspace(handle, dWeights.precision());

        auto dWeightsDescriptor = getPrnnTensorDescriptor(dWeights);

        auto hxDescriptor = getPrnnSingleDescriptor(handle, dWeights.precision());

        PrnnLibrary::prnnRNNBackwardWeights(descriptor.descriptor(),
                                            handle.timesteps,
                                            xDescriptor.descriptors(),
                                            xDescriptor.data(),
                                            hxDescriptor.descriptor(),
                                            nullptr,
                                            yDescriptor.descriptors(),
                                            yDescriptor.data(),
                                            workspace.data(),
                                            workspace.elements() * workspace.precision().size(),
                                            dWeightsDescriptor.descriptor(),
                                            dWeightsDescriptor.data(),
                                            reserve.data(),
                                            reserve.elements() * reserve.precision().size());

    }
    else if(CudnnLibrary::loaded())
    {
        parallel::setNotSynchronized();

        CudnnRNNDescriptor descriptor(handle, dWeights.precision());

        auto yDescriptor = getCudnnTensorDescriptorArray(outputActivations);
        auto xDescriptor = getCudnnTensorDescriptorArray(inputActivations);
        auto workspace   = getWorkspace(handle, dWeights.precision());

        auto dWeightsDescriptor = getCudnnFilterDescriptor(dWeights);

        auto hxDescriptor = getCudnnSingleDescriptor(handle, dWeights.precision());

        CudnnLibrary::cudnnRNNBackwardWeights(descriptor.descriptor(),
                                              handle.timesteps,
                                              xDescriptor.descriptors(),
                                              xDescriptor.data(),
                                              hxDescriptor.descriptor(),
                                              nullptr,
                                              yDescriptor.descriptors(),
                                              yDescriptor.data(),
                                              workspace.data(),
                                              workspace.elements() * workspace.precision().size(),
                                              dWeightsDescriptor.descriptor(),
                                              dWeightsDescriptor.data(),
                                              reserve.data(),
                                              reserve.elements() * reserve.precision().size());

    }
    else
    {
        nativeRNNBackPropGradientsRecurrent(dWeights,
                                            inputActivations,
                                            outputActivations,
                                            reserve,
                                            handle);
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



