#pragma once

// Standard Library Includes
#include <vector>
#include <cstddef>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Operation; } }
namespace lucius { namespace matrix { class Precision; } }

namespace lucius
{
namespace matrix
{

enum RecurrentLayerDirection
{
    RECURRENT_FORWARD,
    RECURRENT_REVERSE,
    RECURRENT_BIDIRECTIONAL
};

enum RecurrentLayerType
{
    RECURRENT_SIMPLE_TYPE,
    RECURRENT_GRU_TYPE,
    RECURRENT_LSTM_TYPE
};

enum RecurrentLayerInputMode
{
    RECURRENT_LINEAR_INPUT,
    RECURRENT_SKIP_INPUT
};

class RecurrentOpsHandle
{
public:
    RecurrentOpsHandle(size_t layerSize, size_t miniBatchSize, size_t timesteps,
        size_t layers = 1,
        RecurrentLayerDirection direction = RECURRENT_FORWARD,
        RecurrentLayerType layerType = RECURRENT_SIMPLE_TYPE,
        RecurrentLayerInputMode inputMode = RECURRENT_SKIP_INPUT);

public:
    std::string toString() const;

public:
    size_t layerSize;
    size_t miniBatchSize;
    size_t timesteps;

public:
    size_t layers;

public:
    RecurrentLayerDirection     direction;
    RecurrentLayerType          layerType;
    RecurrentLayerInputMode     inputMode;
};


matrix::Matrix createReserveRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);

matrix::Matrix createWeightsRecurrent(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
matrix::Matrix sliceLayerWeights(const matrix::Matrix& weights, const RecurrentOpsHandle& handle,
    size_t index);
size_t getTotalWeightMatrices(const RecurrentOpsHandle& handle);

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
                          const RecurrentOpsHandle& handle);

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
    const matrix::Matrix& activations,
    matrix::Matrix& reserve,
    const RecurrentOpsHandle& handle);

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
    const RecurrentOpsHandle& handle);

typedef std::vector<size_t> IndexVector;

void recurrentZeroEnds(Matrix& activations, const IndexVector& lengths);


}
}


