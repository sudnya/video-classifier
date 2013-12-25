/*	\file   CublasMatrix.cpp
	\date   Monday September 2, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the CublasMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CublasMatrix.h>

#include <minerva/matrix/interface/CublasLibrary.h>
#include <minerva/matrix/interface/CudaRuntimeLibrary.h>

#include <minerva/util/interface/Knobs.h>

// Standard Library Includes
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <random>
#include <ctime>

namespace minerva
{

namespace matrix
{

typedef MatrixImplementation Value;

CublasMatrix::CublasMatrix(size_t r, size_t c, const FloatVector& data)
: MatrixImplementation(r, c, data)
{
	resize(rows(), columns());
}

void CublasMatrix::resize(size_t rows, size_t columns)
{
	_data.resize(rows * columns);
	
	_rows	 = rows;
	_columns = columns;
}

Value* CublasMatrix::appendColumns(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);

	assert(empty() || (rows() == m->rows()));

    size_t resultRows = rows();

    if(empty())
    {
        resultRows = m->rows();
    }

	CublasMatrix* result = new CublasMatrix(resultRows,
		columns() + m->columns());
	
	// Copy rows from the original and appended matrices
	for(size_t row = 0; row != resultRows; ++row)
	{
		size_t originalPosition = getPosition(row, 0);
		size_t newPosition = result->getPosition(row, 0);
	
		std::memcpy(&result->_data[newPosition], &_data[originalPosition],
			columns() * sizeof(float));

		size_t appendedPosition = m->getPosition(row, 0);
		newPosition += columns();
		
		std::memcpy(&(*result)._data[newPosition], &m->_data[appendedPosition],
			m->columns() * sizeof(float));
	}
	
	return result;
}

Value* CublasMatrix::appendRows(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);

	assert(empty() || (columns() == m->columns()));

    size_t resultColumns = columns();

    if(empty())
    {
        resultColumns = m->columns();
    }

	CublasMatrix* result = new CublasMatrix(rows() + m->rows(), resultColumns);
	
	// Copy rows from the original and appended matrices
	std::memcpy(&result->_data[0], &_data[0], size() * sizeof(float));
	std::memcpy(&result->_data[size()], &m->_data[0],
		m->size() * sizeof(float));
	
	return result;
}

Value* CublasMatrix::transpose() const
{
	//auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(this != nullptr);

	CublasMatrix* result = nullptr;

	float* a = nullptr;
	float* b = nullptr;
	float* c = nullptr;
	//float* t = nullptr;
	try
	{
		result = new CublasMatrix(this->columns(), this->rows());
		
		a = (float*)CudaRuntimeLibrary::cudaMalloc(sizeof(float) * size()        );
		c = (float*)CudaRuntimeLibrary::cudaMalloc(sizeof(float) * result->size());
		
		CudaRuntimeLibrary::cudaMemcpy(a, &_data[0],    sizeof(float) *    size());
	
		float alpha = 1.0f;
		float beta  = 0.0f;
		
		//lda = num_col_A = num_row_AT = N;
		int lda = columns();

		// ldb = num_col_B = num_row_BT = N;
		int ldb = rows();//this->columns();

		// ldc = num_col_C, num_row_CT = N;
		int ldc = result->columns();

		// m and n in the cuBLAS GEMM routine are the #rows and #cols of the result matrix C,

		// m = num_row_C, num_col_CT
		int m = result->columns();

		// n = num_col_C, num_row_CT
		int n = result->rows();
		
		CublasLibrary::cublasSgeam(CublasLibrary::CUBLAS_OP_T,
			CublasLibrary::CUBLAS_OP_N, m, n, &alpha, a, lda, &beta, b, ldb, c, ldc);
		
		CudaRuntimeLibrary::cudaMemcpy(&result->_data[0], c,
			sizeof(float) * result->size(), CudaRuntimeLibrary::cudaMemcpyDefault);
	}
	catch(...)
	{
		CudaRuntimeLibrary::cudaFree(a);
		CudaRuntimeLibrary::cudaFree(c);
		
		delete result;
		
		throw;
	}
	
	CudaRuntimeLibrary::cudaFree(a);
	CudaRuntimeLibrary::cudaFree(c);
	
	return result;
}
 
Value* CublasMatrix::multiply(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);
	assert(columns() == m->rows());

	CublasMatrix* result = nullptr;

	float* a = nullptr;
	float* b = nullptr;
	float* c = nullptr;
	
	try
	{
		result = new CublasMatrix(rows(), m->columns());
	
		a = (float*)CudaRuntimeLibrary::cudaMalloc(sizeof(float) * size()        );
		b = (float*)CudaRuntimeLibrary::cudaMalloc(sizeof(float) * m->size()     );
		c = (float*)CudaRuntimeLibrary::cudaMalloc(sizeof(float) * result->size());
		
		CudaRuntimeLibrary::cudaMemcpy(a, &_data[0],    sizeof(float) *    size());
		CudaRuntimeLibrary::cudaMemcpy(b, &m->_data[0], sizeof(float) * m->size());
		
		float alpha = 1.0f;
		float beta  = 0.0f;
		
		
		//lda = num_col_A = num_row_AT = N;
		int lda = columns();

		// ldb = num_col_B = num_row_BT = N;
		int ldb = m->columns();

		// ldc = num_col_C = N;
		int ldc = result->columns();

		// m and n in the cuBLAS GEMM routine are the #rows and #cols of
		// the result matrix C,

		// k is the common dimension of A^T and B,

		// k = num_col_AT = num_row_B = M;
		int k = columns();

		// n = num_col_C
		int n = result->rows();

		// m = num_row_C
		int m = result->columns();
		
		CublasLibrary::cublasSgemm(CublasLibrary::CUBLAS_OP_N,
			CublasLibrary::CUBLAS_OP_N, m, n, k, &alpha,
			b, ldb, a, lda, &beta, c, ldc);
		
		/*CublasLibrary::cublasSgemm(CublasLibrary::CUBLAS_OP_T,
			CublasLibrary::CUBLAS_OP_T,
			result->rows(), result->columns(), columns(), &alpha, a, columns(),
			b, m->columns(), &beta, c, result->rows());
		*/
		CudaRuntimeLibrary::cudaMemcpy(&result->_data[0], c,
			sizeof(float) * result->size(), CudaRuntimeLibrary::cudaMemcpyDefault);
	}
	catch(...)
	{
		CudaRuntimeLibrary::cudaFree(a);
		CudaRuntimeLibrary::cudaFree(b);
		CudaRuntimeLibrary::cudaFree(c);
		
		delete result;
		
		throw;
	}
	
	CudaRuntimeLibrary::cudaFree(a);
	CudaRuntimeLibrary::cudaFree(b);
	CudaRuntimeLibrary::cudaFree(c);
	
	//result->transposeSelf();
	
	return result;
}

Value* CublasMatrix::multiply(float f) const
{
	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value *= f;
	}
	
	return result;
}

Value* CublasMatrix::elementMultiply(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);

	assert(m->rows()    == rows()   );
	assert(m->columns() == columns());

    CublasMatrix* result = new CublasMatrix(*this);

	// TODO: faster
	auto rValue = result->_data.begin();
	for(auto value = m->_data.begin(); value != m->_data.end();
		++value, ++rValue)
	{
		*rValue *= *value;
	}

    return result;
}

Value* CublasMatrix::add(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);
	
	assert(m->rows()    == rows());
	assert(m->columns() == columns());

	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	auto rValue = result->_data.begin();
	for(auto value = m->_data.begin(); value != m->_data.end();
		++value, ++rValue)
	{
		*rValue += *value;
	}
	
	return result;
}

Value* CublasMatrix::addBroadcastRow(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);
	
	assert(m->columns() == columns());

	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	size_t columnSize = columns();
	size_t rowSize    = rows();

	for(size_t r = 0; r < rowSize; ++r)
	{
		for(size_t c = 0; c < columnSize; ++c)
		{
			result->data()[result->getPosition(r, c)] =
				data()[getPosition(r, c)] + m->data()[m->getPosition(0, c)];
		}
	}

	return result;
}

Value* CublasMatrix::add(float f) const
{
	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value += f;
	}
	
	return result;
}

Value* CublasMatrix::subtract(const Value* matrix) const
{
	auto m = dynamic_cast<const CublasMatrix*>(matrix);	
	assert(m != nullptr);
	
	assert(m->rows()    == rows());
	assert(m->columns() == columns());

	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	auto rValue = result->_data.begin();
	for(auto value = m->_data.begin(); value != m->_data.end();
		++value, ++rValue)
	{
		*rValue -= *value;
	}
	
	return result;
}

Value* CublasMatrix::subtract(float f) const
{
	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value -= f;
	}
	
	return result;
}

Value* CublasMatrix::slice(size_t startRow, size_t startColumn,
	size_t rows, size_t columns) const
{
	CublasMatrix* result = new CublasMatrix(rows, columns);
	
	assert(startRow    + rows    <= this->rows()   );
	assert(startColumn + columns <= this->columns());
	
	// TODO: faster
	for(size_t row = 0; row != rows; ++row)
	{
		std::memcpy(&result->data()[result->getPosition(row, 0)],
			&data()[getPosition(row + startRow, startColumn)],
			columns * sizeof(float));
	}
	
	return result;
}

Value* CublasMatrix::log() const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->logSelf();

    return result;
}

Value* CublasMatrix::abs() const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->absSelf();

    return result;
}

Value* CublasMatrix::negate() const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->negateSelf();

    return result;
}

Value* CublasMatrix::sigmoid() const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->sigmoidSelf();

    return result;
}

Value* CublasMatrix::sigmoidDerivative() const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->sigmoidDerivativeSelf();

    return result;
}

Value* CublasMatrix::klDivergence(float sparsity) const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->klDivergenceSelf(sparsity);

    return result;
}

Value* CublasMatrix::klDivergenceDerivative(float sparsity) const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->klDivergenceDerivativeSelf(sparsity);

    return result;
}

void CublasMatrix::negateSelf()
{
	for(auto& f : _data)
	{
		f = -f;
	}
}

void CublasMatrix::logSelf()
{
	for(auto& f : _data)
	{
		f = std::log(f);
	}
}

void CublasMatrix::absSelf()
{
	for(auto& f : _data)
	{
		f = std::abs(f);
	}
}

static float sigmoid(float v)
{
    if(v < -50.0f) return 0.0f;
    if(v >  50.0f) return 1.0f;
    
    return 1.0f / (1.0f + std::exp(-v)); 
}

static float sigmoidDerivative(float v)
{
    // f(x) = 1/(1+e^-x)
    // dy/dx = f(x)' = f(x) * (1 - f(x))
	//float element = sigmoid(v) * (1.0f - sigmoid(v));
	
	float element = v * (1.0f - v);
	
	return element;
}

void CublasMatrix::sigmoidSelf()
{
	for(auto& f : _data)
	{
		f = matrix::sigmoid(f);
	}
}

void CublasMatrix::sigmoidDerivativeSelf()
{
	for(auto& f : _data)
	{
		f = matrix::sigmoidDerivative(f);
	}
}

static float klDivergence(float value, float sparsity)
{
	// f(x,y) = y * log(y/x) + (1-y) * log((1 - y)/(1 - x))
	if(value > 1.0f) value = 1.0f;
	if(value < 0.0f) value = 0.0f;

	return sparsity * std::log(sparsity / value) +
		(1.0f - sparsity) * std::log((1.0f - sparsity) / (1.0f - value));
}

static float klDivergenceDerivative(float value, float sparsity)
{
	// f(x,y) = y * log(y/x) + (1-y) * log((1 - y)/(1 - x))
	// dy/dx = f'(x,y) = (-y/x + (1-y)/(1-x))
	if(value > 1.0f) value = 1.0f;
	if(value < 0.0f) value = 0.0f;

	return (-sparsity / value + (1.0f - sparsity)/(1.0f - value));
}

void CublasMatrix::klDivergenceSelf(float sparsity)
{
	for(auto& f : _data)
	{
		f = matrix::klDivergence(f, sparsity);
	}
}

void CublasMatrix::klDivergenceDerivativeSelf(float sparsity)
{
	for(auto& f : _data)
	{
		f = matrix::klDivergenceDerivative(f, sparsity);
	}
}

void CublasMatrix::assignUniformRandomValues(
	std::default_random_engine& generator, float min, float max)
{
	// TODO: use cuRand
	std::uniform_real_distribution<float> distribution(min, max);

	for(auto& f : _data)
	{
		f = distribution(generator);
	}
}

Value* CublasMatrix::greaterThanOrEqual(float f) const
{
	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	for(auto& value : result->_data)
	{
		value = (value >= f) ? 1.0f : 0.0f;
	}
	
	return result;
}

Value* CublasMatrix::equals(const Value* m) const
{
	assert(m->size() == size());

	CublasMatrix* result = new CublasMatrix(*this);
	
	// TODO: faster
	auto value = m->data().begin();
	for(auto resultValue = result->data().begin(); resultValue != result->data().end();
		++resultValue, ++value)
	{
		*resultValue = (*resultValue == *value) ? 1.0f : 0.0f;
	}
	
	return result;
}

void CublasMatrix::transposeSelf()
{
    // TODO: in place
	auto matrix = transpose();
	
	auto cublasMatrix = dynamic_cast<CublasMatrix*>(matrix);
	assert(cublasMatrix != nullptr);
	
	*this = *cublasMatrix;
	
	delete cublasMatrix;
}

float CublasMatrix::reduceSum() const
{
    float sum = 0.0f;

    for(auto& f : _data)
    {
        sum += f;
    }

    return sum;
}

Value* CublasMatrix::reduceSumAlongColumns() const
{
	auto result = new CublasMatrix(rows(), 1);
	
	size_t rowCount    = rows();
	size_t columnCount = columns(); 

	for(size_t row = 0; row < rowCount; ++row)
	{
		float value = 0.0f;
		
		for(size_t column = 0; column < columnCount; ++column)
		{
			value += data()[getPosition(row, column)];
		}
		
		result->data()[result->getPosition(row, 0)] = value;
	}
    
	return result;
}

const CublasMatrix::FloatVector& CublasMatrix::data() const
{
	return _data;
}

CublasMatrix::FloatVector& CublasMatrix::data()
{
	return _data;
}

Value* CublasMatrix::clone() const
{
	return new CublasMatrix(*this);
}

bool CublasMatrix::isSupported()
{
	if(!util::KnobDatabase::getKnobValue("CublasMatrix::Enable", false))
	{
		return false;
	}
	
	CublasLibrary::load();

	return CublasLibrary::loaded();
}

}

}

