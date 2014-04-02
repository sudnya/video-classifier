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
	static void multiply(float* result, const float* left, bool leftTransposed,
		const float* right, bool rightTranposed, size_t blocks, size_t rows,
		size_t columns, size_t rightRows, size_t rightColumns);
	static void multiply(float* result, const float* left, float value, size_t size);
	static void elementMultiply(float* result, const float* left, const float* right, size_t size);

public:
	static void convolutionalMultiply(float* result, const float* left, 
		const float* right, bool rightTransposed, size_t resultBlocks, size_t leftBlocks,
		size_t leftRowsPerBlock, size_t leftColumnsPerBlock, size_t rightBlocks,
		size_t rightRowsPerBlock, size_t rightColumnsPerBlock, size_t step);
	static void reverseConvolutionalMultiply(float* result, const float* left, bool leftTransposed,
		const float* right, bool rightTransposed, size_t resultBlocks, size_t leftBlocks,
		size_t leftRowsPerBlock, size_t leftColumnsPerBlock, size_t rightBlocks,
		size_t rightRowsPerBlock, size_t rightColumnsPerBlock);
	static void convolutionalAddBroadcastRow(float* result, const float* left, const float* right,
		size_t leftBlocks, size_t rightBlocks, size_t rows, size_t columns);
	

public:
	static void add(float* result, const float* left, const float* right, size_t size);
	static void addBroadcastRow(float* result, const float* left, const float* right,
		size_t blocks, size_t rows, size_t columns, bool isRowSparse);
	static void add(float* result, const float* left, float f, size_t size);

public:
	static void subtract(float* result, const float* left, const float* right, size_t size);
	static void subtract(float* result, const float* left, float f, size_t size);

public:
	static void klDivergence(float* result, float f, size_t size);
	static void klDivergenceDerivative(float* result, float f, size_t size);

public:
	static void transpose(float* result, const float* left, size_t blocks, size_t rows, size_t columns);
	static void copy(float* result, const float* left, size_t size);
	static void negate(float* result, const float* left, size_t size);
	static void log(float* result, const float* left, size_t size);
	static void sigmoid(float* result, const float* left, size_t size);
	static void sigmoidDerivative(float* result, const float* left, size_t size);

public:
	static void negateSelf(float* result, size_t size);
	static void logSelf(float* result, size_t size);
	static void sigmoidSelf(float* result, size_t size);
	static void sigmoidDerivativeSelf(float* result, size_t size);

public:
	static void assignUniformRandomValues(float* result, float min, float max,
		size_t size);

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








