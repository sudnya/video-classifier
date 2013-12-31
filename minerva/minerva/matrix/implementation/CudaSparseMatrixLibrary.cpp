/*! \file   CudaSparseMatrixLibrary.cpp
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The source file for the CudaSparseMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CudaSparseMatrixLibrary.h>
#include <minerva/matrix/interface/CudaDriverTypes.h>

namespace minerva
{

namespace matrix
{

class CudaSparseMatrixLibrarySingleton
{

public:
	bool loaded() const;
	void load();

public:
	CUmodule module;
	

};

static std::unique_ptr<CudaSparseMatrixLibrarySingleton> singleton(new CudaSparseMatrixLibrarySingleton);

static void launchKernel(const std::string& name, const ByteVector& parameters)
{
	CUfunction function;
	
	
}

template<typename Args...>
static void launchKernel(const std::string& name, Args...)
{
	
}


void CudaSparseMatrixLibrary::multiply(float* result, const float* left, const float* right,
	size_t blocks, size_t rows, size_t columns, size_t isRowSparse)
{
	// cublas
}

void CudaSparseMatrixLibrary::multiply(float* result, const float* left, float value, size_t size)
{
	launchKernel("multiplyFloat", result, left, value, size);
}

void CudaSparseMatrixLibrary::elementMultiply(float* result, const float* left, const float* right, size_t size)
{
	launchKernel("elementMultiply", result, left, value, size);
}

void CudaSparseMatrixLibrary::add(float* result, const float* left, const float* right, size_t size)
{
	launchKernel("add", left, right, size);
}

void CudaSparseMatrixLibrary::addBroadcastRow(float* result, const float* left, const float* right,
	size_t blocks, size_t rows, size_t columns, size_t isRowSparse)
{
	if(isRowSparse)
	{
		launchKernel("addBroadcastRow", left, right, blocks, rows, columns);
	}
	else
	{
		launchKernel("addBroadcastRowColumnSparse", left, right, blocks, rows, columns);
	}
}

void CudaSparseMatrixLibrary::add(float* result, const float* left, float f, size_t size)
{
	launchKernel("addFloat", result, left, f, size);
}

void CudaSparseMatrixLibrary::subtract(float* result, const float* left, const float* right, size_t size)
{
	launchKernel("subtract", result, left, right, size);
}

void CudaSparseMatrixLibrary::subtract(float* result, const float* left, float f, size_t size)
{
	launchKernel("subtractFloat", result, left, f, size);
}

void CudaSparseMatrixLibrary::klDivergence(float* result, float f, size_t size)
{
	launchKernel("klDivergence", result, f, size);
}

void CudaSparseMatrixLibrary::klDivergenceDerivative(float* result, float f, size_t size)
{
	launchKernel("klDivergenceDerivative", result, f, size);
}

void CudaSparseMatrixLibrary::negate(float* result, size_t size)
{
	launchKernel("negate", result, size);
}

void CudaSparseMatrixLibrary::log(float* result, size_t size)
{
	launchKernel("log", result, size);
}

void CudaSparseMatrixLibrary::sigmoid(float* result, size_t size)
{
	launchKernel("sigmoid", result, size);
}

void CudaSparseMatrixLibrary::sigmoidDerivative(float* result, size_t size)
{
	launchKernel("sigmoidDerivative", result, size);
}

void CudaSparseMatrixLibrary::assignUniformRandomValues(float* result, float min, float max, size_t size)
{
	// curand
}

void CudaSparseMatrixLibrary::greaterThanOrEqual(float* result, const float* left, float f, size_t size)
{
	launchKernel("greaterThanOrEqual", result, left, f, size);
}

void CudaSparseMatrixLibrary::equals(float* result, const float* left, const float* right, size_t size)
{
	launchKernel("equals", result, left, right, size);
}

float CudaSparseMatrixLibrary::reduceSum(const float* input, size_t size)
{
	// cublas
}

void CudaSparseMatrixLibrary::reduceSumAlongRows(float* result, const float* input,
	size_t blocks, size_t rows, size_t columns, size_t isRowSparse)
{
	if(isRowSparse)
	{
		launchKernel("reduceSumAlongRows", result, input, blocks, rows, columns);
	}
	else
	{
		launchKernel("reduceSumAlongRowsColumnSparse", result, input, blocks, rows, columns);
	}
}

void CudaSparseMatrixLibrary::reduceSumAlongColumns(float* result, const float* input,
	size_t blocks, size_t rows, size_t columns, size_t isRowSparse)
{
	if(isRowSparse)
	{
		launchKernel("reduceSumAlongColumns", result, input, blocks, rows, columns);
	}
	else
	{
		launchKernel("reduceSumAlongColumnsColumnSparse", result, input, blocks, rows, columns);
	}
}

void CudaSparseMatrixLibrary::load()
{
	CudaDriver::load();
	
	singleton->load();
	
	CublasLibrary::load();
}

bool CudaSparseMatrixLibrary::loaded()
{
	return singleton->loaded() && CudaDriver::loaded() && CublasLibrary::loaded();
}

}

}









