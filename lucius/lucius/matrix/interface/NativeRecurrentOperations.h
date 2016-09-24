#pragma once

// Standard Library Includes
#include <vector>
#include <cstddef>
#include <tuple>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;             } }
namespace lucius { namespace matrix { class Operation;          } }
namespace lucius { namespace matrix { class Precision;          } }
namespace lucius { namespace matrix { class RecurrentOpsHandle; } }

namespace lucius
{

namespace matrix
{

size_t nativeRNNGetTrainingReserveSize(const RecurrentOpsHandle& handle,
    const Precision& precision);

size_t nativeRNNGetWeightsSize(const RecurrentOpsHandle& handle,
    const Precision& precision);

std::tuple<size_t, size_t> nativeRNNGetLinearLayerMatrixSizeAndOffset(
    const RecurrentOpsHandle& handle, const Precision& precision,
    size_t layer, size_t offsetInLayer);
std::tuple<size_t, size_t> nativeRNNGetBiasLayerMatrixSizeAndOffset(
    const RecurrentOpsHandle& handle, const Precision& precision,
    size_t layer, size_t offsetInLayer);

void nativeRNNForwardPropRecurrent(matrix::Matrix& outputActivations,
                                   const matrix::Matrix& inputActivations,
                                   matrix::Matrix& reserve,
                                   const matrix::Matrix& weights,
                                   const RecurrentOpsHandle& handle);

void nativeRNNBackPropDeltasRecurrent(matrix::Matrix& inputDeltas,
                                      const matrix::Matrix& outputDeltas,
                                      const matrix::Matrix& weights,
                                      const matrix::Matrix& outputActivations,
                                      matrix::Matrix& reserve,
                                      const RecurrentOpsHandle& handle);

void nativeRNNBackPropGradientsRecurrent(matrix::Matrix& dWeights,
                                         const matrix::Matrix& inputActivations,
                                         const matrix::Matrix& outputActivations,
                                         const matrix::Matrix& reserve,
                                         const RecurrentOpsHandle& handle);

}

}

