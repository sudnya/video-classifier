/*	\file   CublasMatrix.cpp
	\date   Monday September 2, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the CublasMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CublasMatrix.h>

#include <minerva/matrix/interface/CublasLibrary.h>

// Standard Library Includes
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace minerva
{

namespace matrix
{

typedef MatrixImplementation Value;

CublasMatrix::CublasMatrix(size_t r, size_t c, const FloatVector& data)
: MatrixImplementation(r, c), _data(data)
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
		size_t originalPosition = _getPosition(row, 0);
		size_t newPosition = result->_getPosition(row, 0);
	
		std::memcpy(&result->_data[newPosition], &_data[originalPosition],
			columns() * sizeof(float));

		size_t appendedPosition = m->_getPosition(row, 0);
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
		
		a = (float*)CublasLibrary::cudaMalloc(sizeof(float) * size()        );
		//b = (float*)CublasLibrary::cudaMalloc(sizeof(float) * this->size()     );
		c = (float*)CublasLibrary::cudaMalloc(sizeof(float) * result->size());
		
		CublasLibrary::cudaMemcpy(a, &_data[0],    sizeof(float) *    size());
		//CublasLibrary::cudaMemcpy(b, &this->_data[0], sizeof(float) * this->size());
		
	
		float alpha = 1.0f;
		float beta  = 0.0f;
		
		
		//lda = num_col_A = num_row_AT = N;
		int lda = columns();

		// ldb = num_col_B = num_row_BT = N;
		int ldb = this->columns();

		// ldc = num_col_C = N;
		int ldc = result->rows();

		// m and n in the cuBLAS GEMM routine are the #rows and #cols of the result matrix C,

		// n = num_col_C
		int n = result->rows();

		// m = num_row_C
		int m = result->columns();
		
		CublasLibrary::cublasSgeam(CublasLibrary::CUBLAS_OP_N,
			CublasLibrary::CUBLAS_OP_N, m, n, &alpha, a, lda, &beta, b, ldb, c, ldc);
		
		CublasLibrary::cudaMemcpy(&result->_data[0], c,
			sizeof(float) * result->size(), CublasLibrary::cudaMemcpyDefault);
	}
	catch(...)
	{
		CublasLibrary::cudaFree(a);
		CublasLibrary::cudaFree(b);
		CublasLibrary::cudaFree(c);
		
		delete result;
		
		throw;
	}
	
	CublasLibrary::cudaFree(a);
	CublasLibrary::cudaFree(b);
	CublasLibrary::cudaFree(c);
	
	//result->transposeSelf();
	
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
	
		a = (float*)CublasLibrary::cudaMalloc(sizeof(float) * size()        );
		b = (float*)CublasLibrary::cudaMalloc(sizeof(float) * m->size()     );
		c = (float*)CublasLibrary::cudaMalloc(sizeof(float) * result->size());
		
		CublasLibrary::cudaMemcpy(a, &_data[0],    sizeof(float) *    size());
		CublasLibrary::cudaMemcpy(b, &m->_data[0], sizeof(float) * m->size());
		
		float alpha = 1.0f;
		float beta  = 0.0f;
		
		
		//lda = num_col_A = num_row_AT = N;
		int lda = columns();

		// ldb = num_col_B = num_row_BT = N;
		int ldb = m->columns();

		// ldc = num_col_C = N;
		int ldc = result->columns();

		// m and n in the cuBLAS GEMM routine are the #rows and #cols of the result matrix C,

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
		CublasLibrary::cudaMemcpy(&result->_data[0], c,
			sizeof(float) * result->size(), CublasLibrary::cudaMemcpyDefault);
	}
	catch(...)
	{
		CublasLibrary::cudaFree(a);
		CublasLibrary::cudaFree(b);
		CublasLibrary::cudaFree(c);
		
		delete result;
		
		throw;
	}
	
	CublasLibrary::cudaFree(a);
	CublasLibrary::cudaFree(b);
	CublasLibrary::cudaFree(c);
	
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
		for(size_t column = 0; column != columns; ++column)
		{
			result->setValue(row, column,
				getValue(row + startRow, column + startColumn));
		}
	}
	
	return result;
}

Value* CublasMatrix::log() const
{
    CublasMatrix* result = new CublasMatrix(*this);
	
	result->logSelf();

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

static float sigmoid(float v)
{
    if(v < -50.0f) return 0.0f;
    if(v >  50.0f) return 1.0f;
    
    return 1.0f / (1.0f + std::exp(-v)); 
}

void CublasMatrix::sigmoidSelf()
{
	for(auto& f : _data)
	{
		f = matrix::sigmoid(f);
	}
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

CublasMatrix::FloatVector CublasMatrix::data() const
{
	return _data;
}

void CublasMatrix::setDataRowMajor(const FloatVector& data)
{
	assert(data.size() == size());
	
	_data = data;
}

void CublasMatrix::setValue(size_t row, size_t column, float value)
{
	assert(row < rows());
	assert(column < columns());

	size_t position = _getPosition(row, column);
	
	_data[position] = value;
}

float CublasMatrix::getValue(size_t row, size_t column) const
{
	assert(row < rows());
	assert(column < columns());

	size_t position = _getPosition(row, column);
	
	return _data[position];
}

Value* CublasMatrix::clone() const
{
	return new CublasMatrix(*this);
}

bool CublasMatrix::isSupported()
{
	CublasLibrary::load();

	return CublasLibrary::loaded();
}

size_t CublasMatrix::_getPosition(size_t row, size_t column) const
{
	return row * columns() + column;
}

}

}

