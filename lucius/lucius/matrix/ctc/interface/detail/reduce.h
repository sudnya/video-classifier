#pragma once
namespace lucius
{
namespace matrix
{
namespace ctc
{

ctcStatus_t reduce_negate(const float* input, float* output, int rows, int cols, bool axis, cudaStream_t stream);
ctcStatus_t reduce_exp(const float* input, float* output, int rows, int cols, bool axis, cudaStream_t stream);
ctcStatus_t reduce_max(const float* input, float* output, int rows, int cols, bool axis, cudaStream_t stream);
}
}
}
