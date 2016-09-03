
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

CudnnTensorDescriptor getCudnnTensorDescriptor(const matrix::Matrix& input)
{
    return CudnnTensorDescriptor(input);
}

CudnnTensorDescriptor getCudnnTensorDescriptor(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    auto size = Dimension(handle.layerSize, handle.miniBatchSize, handle.timesteps);

    return CudnnTensorDescriptor(size, linearStride(size), precision);
}

PrnnTensorDescriptor getPrnnTensorDescriptor()
{
    return PrnnTensorDescriptor({});
}

PrnnTensorDescriptor getPrnnTensorDescriptor(const matrix::Matrix& input)
{
    return PrnnTensorDescriptor(input);
}

PrnnTensorDescriptor getPrnnTensorDescriptor(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    auto size = Dimension(handle.layerSize, handle.miniBatchSize, handle.timesteps);

    return PrnnTensorDescriptor(size, linearStride(size), precision);
}

CudnnFilterDescriptor getCudnnFilterDescriptor(const matrix::Matrix& weights)
{
    return CudnnFilterDescriptor(weights);
}

CudnnFilterDescriptor getCudnnFilterDescriptor()
{
    return CudnnFilterDescriptor({});
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
        PrnnTensorDescriptor inputDescriptor = getPrnnTensorDescriptor(handle, precision);

        PrnnLibrary::prnnGetRNNTrainingReserveSize(descriptor.descriptor(),
                                                   handle.timesteps,
                                                   &inputDescriptor.descriptor(),
                                                   &reserveSize);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle, precision);
        CudnnTensorDescriptor inputDescriptor = getCudnnTensorDescriptor(handle, precision);

        CudnnLibrary::cudnnGetRNNTrainingReserveSize(descriptor.descriptor(),
                                                     handle.timesteps,
                                                     &inputDescriptor.descriptor(),
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
        PrnnTensorDescriptor inputDescriptor = getPrnnTensorDescriptor(handle, precision);

        PrnnLibrary::prnnGetRNNParamsSize(descriptor.descriptor(),
                                          inputDescriptor.descriptor(),
                                          &weightsSize);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle, precision);
        CudnnTensorDescriptor inputDescriptor = getCudnnTensorDescriptor(handle, precision);

        CudnnLibrary::cudnnGetRNNParamsSize(descriptor.descriptor(),
                                            inputDescriptor.descriptor(),
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
        auto inputDescriptor = getPrnnTensorDescriptor(handle, weights.precision());
        auto weightDescriptor = getPrnnTensorDescriptor(weights);
        PrnnTensorDescriptor filterDescriptor = getPrnnTensorDescriptor();

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
        CudnnRNNDescriptor descriptor(handle, weights.precision());
        auto inputDescriptor = getCudnnTensorDescriptor(handle, weights.precision());
        auto weightDescriptor = getCudnnFilterDescriptor(weights);
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
        std::tie(size, offset) = nativeRNNGetLinearLayerMatrixSizeAndOffset(
            handle, weights.precision(), layer, offsetInLayer);
    }

    offset /= weights.precision().size();
    size   /= weights.precision().size();

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
        auto inputDescriptor = getPrnnTensorDescriptor(handle, weights.precision());
        auto weightDescriptor = getPrnnTensorDescriptor(weights);
        PrnnTensorDescriptor filterDescriptor = getPrnnTensorDescriptor();

        PrnnLibrary::prnnGetRNNLinLayerBiasParams(descriptor.descriptor(),
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
        CudnnRNNDescriptor descriptor(handle, weights.precision());
        auto inputDescriptor = getCudnnTensorDescriptor(handle, weights.precision());
        auto weightDescriptor = getCudnnFilterDescriptor(weights);
        CudnnFilterDescriptor filterDescriptor = getCudnnFilterDescriptor();

        CudnnLibrary::cudnnGetRNNLinLayerBiasParams(descriptor.descriptor(),
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
        std::tie(size, offset) = nativeRNNGetBiasLayerMatrixSizeAndOffset(
            handle, weights.precision(), layer, offsetInLayer);
    }

    offset /= weights.precision().size();
    size   /= weights.precision().size();

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

        auto inputDescriptor = getPrnnTensorDescriptor(handle, precision);

        PrnnLibrary::prnnGetRNNWorkspaceSize(descriptor.descriptor(),
                                             handle.timesteps,
                                             &inputDescriptor.descriptor(),
                                             &size);
    }
    else if(CudnnLibrary::loaded())
    {
        CudnnRNNDescriptor descriptor(handle, precision);

        auto inputDescriptor = getCudnnTensorDescriptor(handle, precision);

        CudnnLibrary::cudnnGetRNNWorkspaceSize(descriptor.descriptor(),
                                               handle.timesteps,
                                               &inputDescriptor.descriptor(),
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

        auto xDescriptor = getPrnnTensorDescriptor(inputActivations);
        auto yDescriptor = getPrnnTensorDescriptor(outputActivations);
        auto workspace   = getWorkspace(handle, weights.precision());

        auto weightsDescriptor = getPrnnTensorDescriptor(weights);

        auto hxDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto cxDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto hyDescriptor = getPrnnSingleDescriptor(handle, weights.precision());
        auto cyDescriptor = getPrnnSingleDescriptor(handle, weights.precision());

        PrnnLibrary::prnnRNNForward(descriptor.descriptor(),
                                    handle.timesteps,
                                    &xDescriptor.descriptor(),
                                    xDescriptor.data(),
                                    hxDescriptor.descriptor(),
                                    nullptr,
                                    cxDescriptor.descriptor(),
                                    nullptr,
                                    weightsDescriptor.descriptor(),
                                    nullptr,
                                    &yDescriptor.descriptor(),
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

        auto xDescriptor = getCudnnTensorDescriptor(inputActivations);
        auto yDescriptor = getCudnnTensorDescriptor(outputActivations);
        auto workspace   = getWorkspace(handle, weights.precision());

        auto weightsDescriptor = getCudnnFilterDescriptor(weights);

        auto hxDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto cxDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto hyDescriptor = getCudnnSingleDescriptor(handle, weights.precision());
        auto cyDescriptor = getCudnnSingleDescriptor(handle, weights.precision());

        CudnnLibrary::cudnnRNNForwardTraining(descriptor.descriptor(),
                                              handle.timesteps,
                                              &xDescriptor.descriptor(),
                                              xDescriptor.data(),
                                              hxDescriptor.descriptor(),
                                              nullptr,
                                              cxDescriptor.descriptor(),
                                              nullptr,
                                              weightsDescriptor.descriptor(),
                                              nullptr,
                                              &yDescriptor.descriptor(),
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

        auto yDescriptor  = getPrnnTensorDescriptor(outputActivations);
        auto dyDescriptor = getPrnnTensorDescriptor(outputDeltas);
        auto dxDescriptor = getPrnnTensorDescriptor(inputDeltas);
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
                                         &yDescriptor.descriptor(),
                                         yDescriptor.data(),
                                         &dyDescriptor.descriptor(),
                                         dyDescriptor.data(),
                                         dhyDescriptor.descriptor(),
                                         nullptr,
                                         dcyDescriptor.descriptor(),
                                         nullptr,
                                         weightsDescriptor.descriptor(),
                                         weightsDescriptor.data(),
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
        parallel::setNotSynchronized();

        CudnnRNNDescriptor descriptor(handle, weights.precision());

        auto yDescriptor  = getCudnnTensorDescriptor(outputActivations);
        auto dyDescriptor = getCudnnTensorDescriptor(outputDeltas);
        auto dxDescriptor = getCudnnTensorDescriptor(inputDeltas);
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
                                           &yDescriptor.descriptor(),
                                           yDescriptor.data(),
                                           &dyDescriptor.descriptor(),
                                           dyDescriptor.data(),
                                           dhyDescriptor.descriptor(),
                                           nullptr,
                                           dcyDescriptor.descriptor(),
                                           nullptr,
                                           weightsDescriptor.descriptor(),
                                           weightsDescriptor.data(),
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

        auto yDescriptor = getPrnnTensorDescriptor(outputActivations);
        auto xDescriptor = getPrnnTensorDescriptor(inputActivations);
        auto workspace   = getWorkspace(handle, dWeights.precision());

        auto dWeightsDescriptor = getPrnnTensorDescriptor(dWeights);

        auto hxDescriptor = getPrnnSingleDescriptor(handle, dWeights.precision());

        PrnnLibrary::prnnRNNBackwardWeights(descriptor.descriptor(),
                                            handle.timesteps,
                                            &xDescriptor.descriptor(),
                                            xDescriptor.data(),
                                            hxDescriptor.descriptor(),
                                            hxDescriptor.data(),
                                            &yDescriptor.descriptor(),
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

        auto yDescriptor = getCudnnTensorDescriptor(outputActivations);
        auto xDescriptor = getCudnnTensorDescriptor(inputActivations);
        auto workspace   = getWorkspace(handle, dWeights.precision());

        auto dWeightsDescriptor = getCudnnFilterDescriptor(dWeights);

        auto hxDescriptor = getCudnnSingleDescriptor(handle, dWeights.precision());

        CudnnLibrary::cudnnRNNBackwardWeights(descriptor.descriptor(),
                                              handle.timesteps,
                                              &xDescriptor.descriptor(),
                                              xDescriptor.data(),
                                              hxDescriptor.descriptor(),
                                              hxDescriptor.data(),
                                              &yDescriptor.descriptor(),
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



