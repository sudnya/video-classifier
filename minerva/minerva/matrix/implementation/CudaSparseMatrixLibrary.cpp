/*! \file   CudaSparseMatrixLibrary.cpp
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The source file for the CudaSparseMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CudaSparseMatrixLibrary.h>
#include <minerva/matrix/interface/CudaSparseMatrixLibraryPTX.h>
#include <minerva/matrix/interface/CudaDriverTypes.h>
#include <minerva/matrix/interface/CudaDriver.h>
#include <minerva/matrix/interface/CublasLibrary.h>
#include <minerva/matrix/interface/CurandLibrary.h>

#include <minerva/util/interface/debug.h>
#include <minerva/util/interface/Casts.h>

// Standard Library Includes
#include <vector>

namespace minerva
{

namespace matrix
{

class CudaSparseMatrixLibrarySingleton
{
public:
	CudaSparseMatrixLibrarySingleton()
	: isLoaded(false)
	{
		
	}


public:
	bool loaded() const
	{
		return isLoaded;
	}
	
	void load()
	{
		CudaDriver::load();
		
		if(!CudaDriver::loaded()) return;
		
		CudaDriver::cuCtxCreate(&context, 0, 0);
		
		loadModule();
	}

	void loadModule()
	{
		util::log("CudaSparseMatrixLibrary") << "Loading module from binary data.\n";

		CUjit_option options[] = {
			//      CU_JIT_TARGET,
			CU_JIT_ERROR_LOG_BUFFER, 
			CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, 
		};

		const uint32_t errorLogSize       = 2048;
		uint32_t       errorLogActualSize = errorLogSize - 1;

		uint8_t errorLogBuffer[errorLogSize];

		std::memset(errorLogBuffer, 0, errorLogSize);

		void* optionValues[] = {
			//      (void*)CU_TARGET_COMPUTE_20,
			(void*)errorLogBuffer, 
			util::bit_cast<void*>(errorLogActualSize), 
		};

		try
		{
			CudaDriver::cuModuleLoadDataEx(&module, getCudaSparseMatrixLibraryPtx(),
				2, options, optionValues);
		}
		catch(const std::exception& e)
		{

			util::log("CudaSparseMatrixLibrary") << "Binary is:" << getCudaSparseMatrixLibraryPtx() << "\n";

			throw std::runtime_error("Failed to load binary data:\n\tMessage: " +
				std::string((char*)errorLogBuffer));
		}
	}

public:
	CUcontext context;
	CUmodule  module;
	bool      isLoaded;

};

static std::unique_ptr<CudaSparseMatrixLibrarySingleton> singleton(new CudaSparseMatrixLibrarySingleton);

static int getGoodCtaCount()
{
	int processors = 0;

	CudaDriver::cuDeviceGetAttribute(&processors,
		CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);

	return processors * 8;
}

typedef std::vector<uint8_t> ByteVector;

static void launchKernel(const std::string& name, const ByteVector& parameters)
{
	assert(singleton->loaded());
	
	CUfunction function;
	
	// get the function
	CudaDriver::cuModuleGetFunction(&function, singleton->module, name.c_str());
	
	// setup the parameters
	CudaDriver::cuParamSetSize(function, parameters.size());
	CudaDriver::cuParamSetv(function, 0, (void*)parameters.data(), parameters.size());
	
	// set the shape
	CudaDriver::cuFuncSetBlockShape(function, 1, 1, 256);
	CudaDriver::cuFuncSetSharedSize(function, 0);

	// launch it
	CudaDriver::cuLaunchGrid(function, 1, getGoodCtaCount());
}

/*
static void launchKernel(const std::string& name)
{
	launchKernel(name, ByteVector());
}
*/

void align(ByteVector& parameters, size_t offset)
{
	size_t remainder = parameters.size() % offset;
	size_t delta = remainder == 0 ? 0 : offset - remainder;

	for(size_t i = 0; i < delta; ++i)
	{
		parameters.push_back(0);
	}
}

template<typename T>
static void flatten(ByteVector& parameters, const T& value)
{
	align(parameters, sizeof(T));
	parameters.insert(parameters.end(), (uint8_t*)&value,
		((uint8_t*)&value) + sizeof(T));
}

template<typename T, typename... Args>
static void flatten(ByteVector& parameters, const T& value, Args... arguments)
{
	flatten(parameters, value);
	flatten(parameters, arguments...);
}

template<typename... Args>
static void launchKernel(const std::string& name, Args... arguments)
{
	ByteVector parameters;
	
	flatten(parameters, arguments...);
	
	launchKernel(name, parameters);
}

typedef std::vector<float*> FloatPointerVector;

static float* getPointerArray(FloatPointerVector& data, const float* value, size_t blocks, size_t rows, size_t columns)
{
	data.resize(blocks);
	
	size_t position = 0;
	
	for(auto& pointer : data)
	{
		pointer = const_cast<float*>(value + position);
		
		position += rows * columns;
	}
	
	CudaDriver::cuMemHostRegister(data.data(), sizeof(float), CU_MEMHOSTREGISTER_DEVICEMAP);
	
	float* devicePointer = nullptr;
	
	CudaDriver::cuMemHostGetDevicePointer((CUdeviceptr*)&devicePointer, data.data(), 0);
	
	return devicePointer;	
}

static void freePointerArray(FloatPointerVector& value)
{
	CudaDriver::cuMemHostUnregister(value.data());
}

void CudaSparseMatrixLibrary::multiply(float* result, const float* left, const float* right,
	size_t blocks, size_t rows, size_t columns, size_t rightRows, size_t rightColumns)
{
	float alpha = 1.0f;
	float beta  = 0.0f;
	
	//lda = num_col_A = num_row_AT = N;
	int lda = columns;

	// ldb = num_col_B = num_row_BT = N;
	int ldb = rightColumns; 

	// ldc = num_col_C = N;
	int ldc = rightColumns;

	// m and n in the cuBLAS GEMM routine are the #rows and #cols of
	// the result matrix C,

	// k is the common dimension of A^T and B,

	// k = num_col_AT = num_row_B = M;
	int k = columns;

	// n = num_col_C
	int n = rows;

	// m = num_row_C
	int m = rightColumns;

	FloatPointerVector aData;
	FloatPointerVector bData;
	FloatPointerVector cData;
		
	float* a = getPointerArray(aData, left,   blocks, rows, columns);
	float* b = getPointerArray(bData, right,  blocks, rightRows, rightColumns);
	float* c = getPointerArray(cData, result, blocks, rows, rightColumns);

	CublasLibrary::cublasSgemmBatched(CublasLibrary::CUBLAS_OP_N,
		CublasLibrary::CUBLAS_OP_N, m, n, k, &alpha,
		b, ldb, a, lda, &beta, c, ldc, blocks);

	freePointerArray(aData);
	freePointerArray(bData);
	freePointerArray(cData);
}

void CudaSparseMatrixLibrary::multiply(float* result, const float* left, float value, size_t size)
{
	launchKernel("multiplyFloat", result, left, value, size);
}

void CudaSparseMatrixLibrary::elementMultiply(float* result, const float* left, const float* right, size_t size)
{
	launchKernel("elementMultiply", result, left, right, size);
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
	CurandLibrary::curandGenerator_t generator;
	
	CurandLibrary::curandCreateGenerator(&generator, CurandLibrary::CURAND_RNG_PSEUDO_DEFAULT);
	
	CurandLibrary::curandGenerateUniform(generator, result, size);

	if(min != 0.0f || max != 1.0f)
	{
		launchKernel("scaleRandom", result, min, max, size);
	}

	CurandLibrary::curandDestroyGenerator(generator);
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
	float result = 0.0f;

	CudaDriver::cuMemHostRegister(&result, sizeof(float), CU_MEMHOSTREGISTER_DEVICEMAP);
	
	float* devicePointer = nullptr;
	
	CudaDriver::cuMemHostGetDevicePointer((CUdeviceptr*)&devicePointer, &result, 0);
	
	launchKernel("reduceSum", devicePointer, input, size);
	
	CudaDriver::cuMemHostUnregister(&result);
	
	return result;
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
	
	CurandLibrary::load();
}

bool CudaSparseMatrixLibrary::loaded()
{
	return singleton->loaded() && CudaDriver::loaded() &&
		CublasLibrary::loaded() && CurandLibrary::loaded();
}

}

}









