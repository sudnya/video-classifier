
#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/parallel/interface/cuda.h>

namespace minerva
{
namespace matrix
{

CUDA_DECORATOR Dimension linearStride(const Dimension& );
CUDA_DECORATOR Dimension zeros(const Dimension& );
CUDA_DECORATOR Dimension ones(const Dimension& );
CUDA_DECORATOR Dimension removeDimensions(const Dimension& base, const Dimension& toRemove);
CUDA_DECORATOR Dimension intersection(const Dimension& base, const Dimension& toRemove);
CUDA_DECORATOR size_t dotProduct(const Dimension& left, const Dimension& right);
CUDA_DECORATOR Dimension linearToDimension(size_t linearIndex, const Dimension& size);
CUDA_DECORATOR Dimension selectNamedDimensions(const Dimension& selectedDimensions, const Dimension& left, const Dimension& right);

CUDA_DECORATOR void* getAddress(const Dimension& stride, const Dimension& position, void* data, const Precision& precision);
CUDA_DECORATOR const void* getAddress(const Dimension& stride, const Dimension& position, const void* data, const Precision& precision);

}
}

#include <minerva/matrix/implementation/DimensionTransformations.inl>

