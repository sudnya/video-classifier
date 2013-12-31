/*! \file   CudaSparseMatrixLibrary.h
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The header file for the CudaSparseMatrix class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>

namespace minerva
{

namespace matrix
{

/*! \brief An interface to CUDA kernels for sparse matrix operations. */
class CudaSparseMatrixLibrary
{
public:
	static void multiply(float* result, const float* left, const float* right,
		size_t blocks, size_t rows, size_t columns, size_t isRowSparse);
	static void multiply(float* result, const float* left, float value, size_t size);
	static void elementMultiply(float* result, const float* left, const float* right, size_t size);

public:
	static void add(float* result, const float* left, const float* right, size_t size);
	static void addBroadcastRow(float* result, const float* left, const float* right,
		size_t blocks, size_t rows, size_t columns, size_t isRowSparse);
	static void add(float* result, const float* left, float f, size_t size);

public:
	static void subtract(float* result, const float* left, const float* right, size_t size);
	static void subtract(float* result, const float* left, float f, size_t size);

public:
	static void klDivergence(float* result, float f, size_t size);
	static void klDivergenceDerivative(float* result, float f, size_t size);

public:
	static void negate(float* result, size_t size);
	static void log(float* result, size_t size);
	static void sigmoid(float* result, size_t size);
	static void sigmoidDerivative(float* result, size_t size);

public:
	static void assignUniformRandomValues(float* result, float min, float max, size_t size);

public:
	static void greaterThanOrEqual(float* result, const float* left, float f, size_t size);
	static void equals(float* result, const float* left, const float* right, size_t size);

public:
	static float reduceSum(const float* input, size_t size);
	
	static void reduceSumAlongRows(float* result, const float* input,
		size_t blocks, size_t rows, size_t columns, size_t isRowSparse);
	static void reduceSumAlongColumns(float* result, const float* input,
		size_t blocks, size_t rows, size_t columns, size_t isRowSparse);

public:
	static void load();
	static bool loaded();

};

}

}








