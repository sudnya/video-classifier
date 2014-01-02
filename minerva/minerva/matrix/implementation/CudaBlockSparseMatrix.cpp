/*! \file   CudaBlockSparseMatrix.h
	\author Gregory Diamos
	\date   Sunday December 29, 2013
	\brief  The header file for the CudaBlockSparseMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/CudaBlockSparseMatrix.h>

#include <minerva/matrix/interface/BlockSparseMatrixImplementation.h>
#include <minerva/matrix/interface/CudaSparseMatrixLibrary.h>
#include <minerva/matrix/interface/CudaBlockSparseCache.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace matrix
{

typedef CudaBlockSparseMatrix::Value Value;
typedef CudaBlockSparseMatrix::MatrixVector MatrixVector;

CudaBlockSparseMatrix::CudaBlockSparseMatrix(size_t blocks, size_t rows,
	size_t columns, bool rowSparse)
: BlockSparseMatrixImplementation(blocks, rows, columns, rowSparse), _isTransposed(false)
{

}

CudaBlockSparseMatrix::CudaBlockSparseMatrix(const CudaBlockSparseMatrix& m, bool copyData)
: BlockSparseMatrixImplementation(m.blocks(), m.rowsPerBlock(), m.columnsPerBlock(), m.isRowSparse()), _isTransposed(m._isTransposed)
{
	if(copyData)
	{
		auto copy = m.begin();
		for(auto matrix = _matrices.begin(); matrix != _matrices.end(); ++matrix, ++copy)
		{
			*matrix = *copy;
		}
	}
}

CudaBlockSparseMatrix::CudaBlockSparseMatrix(const CudaBlockSparseMatrix& m)
: BlockSparseMatrixImplementation(m.blocks(), m.rowsPerBlock(), m.columnsPerBlock(), m.isRowSparse()), _isTransposed(m._isTransposed)
{
	auto copy = m.begin();
	for(auto matrix = _matrices.begin(); matrix != _matrices.end(); ++matrix, ++copy)
	{
		*matrix = *copy;
	}
}

Value* CudaBlockSparseMatrix::multiply(const Value* m) const
{
	assert(m->blocks() == blocks());

	auto result = new CudaBlockSparseMatrix(blocks(), rowsPerBlock(), m->columnsPerBlock(), isRowSparse());

	auto matrixPointer = _cache->acquire(m);
	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::multiply(resultPointer, devicePointer, matrixPointer,
		blocks(), rowsPerBlock(), columnsPerBlock(), m->rowsPerBlock(), m->columnsPerBlock());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;
}

Value* CudaBlockSparseMatrix::multiply(float f) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::multiply(resultPointer, devicePointer, f, size());

	_cache->release(this);
	_cache->release(result);
	
	return result;
}

Value* CudaBlockSparseMatrix::elementMultiply(const Value* m) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());

	auto matrixPointer = _cache->acquire(m);
	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::elementMultiply(resultPointer, devicePointer, matrixPointer, size());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;
}

Value* CudaBlockSparseMatrix::add(const Value* m) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());

	auto matrixPointer = _cache->acquire(m);
	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::add(resultPointer, devicePointer, matrixPointer, size());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;
}

Value* CudaBlockSparseMatrix::addBroadcastRow(const Value* m) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	auto matrixPointer = _cache->acquire(m);
	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::addBroadcastRow(resultPointer, devicePointer, matrixPointer,
		blocks(), rowsPerBlock(), columnsPerBlock(), isRowSparse());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;
}

Value* CudaBlockSparseMatrix::add(float f) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::add(resultPointer, devicePointer, f, size());

	_cache->release(this);
	_cache->release(result);
	
	return result;
}

Value* CudaBlockSparseMatrix::subtract(const Value* m) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());
	
	auto matrixPointer = _cache->acquire(m);
	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::subtract(resultPointer, devicePointer, matrixPointer, size());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;

}
Value* CudaBlockSparseMatrix::subtract(float f) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	auto devicePointer = _cache->acquire(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::subtract(resultPointer, devicePointer, f, size());

	_cache->release(this);
	_cache->release(result);
	
	return result;
}

Value* CudaBlockSparseMatrix::log() const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	result->logSelf();
	
	return result;
}

Value* CudaBlockSparseMatrix::negate() const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	result->negateSelf();
	
	return result;
}

Value* CudaBlockSparseMatrix::sigmoid() const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	result->sigmoidSelf();
	
	return result;
}

Value* CudaBlockSparseMatrix::sigmoidDerivative() const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	result->sigmoidDerivativeSelf();
	
	return result;
}

Value* CudaBlockSparseMatrix::klDivergence(float sparsity) const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	auto resultPointer = _cache->acquire(result);
	
	CudaSparseMatrixLibrary::klDivergence(resultPointer, sparsity, size());

	_cache->release(result);
	
	return result;
}

Value* CudaBlockSparseMatrix::klDivergenceDerivative(float sparsity) const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	auto resultPointer = _cache->acquire(result);
	
	CudaSparseMatrixLibrary::klDivergenceDerivative(resultPointer, sparsity, size());

	_cache->release(result);
	
	return result;
}

Value* CudaBlockSparseMatrix::transpose() const
{
	auto result = new CudaBlockSparseMatrix(*this);
	
	result->transposeSelf();
	
	return result;
}

void CudaBlockSparseMatrix::negateSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::negate(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::logSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::log(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::sigmoidSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::sigmoid(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::sigmoidDerivativeSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::sigmoidDerivative(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::transposeSelf()
{
	_isTransposed = !_isTransposed;
}

void CudaBlockSparseMatrix::assignUniformRandomValues(
	std::default_random_engine& engine, float min, float max)
{
	#if 1
	auto devicePointer = _cache->acquireClobber(this);
	
	CudaSparseMatrixLibrary::assignUniformRandomValues(devicePointer,
		min, max, size());
	
	_cache->release(this);
	#else
	
	for(auto& matrix : *this)
	{
		matrix.assignUniformRandomValues(engine, min, max);
	}
	#endif
}

Value* CudaBlockSparseMatrix::greaterThanOrEqual(float f) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);
	
	auto resultPointer = _cache->acquireClobber(this);
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::greaterThanOrEqual(resultPointer, devicePointer, f, size());
	
	_cache->release(result);
	_cache->release(this);
	
	return result;
}

Value* CudaBlockSparseMatrix::equals(const Value* m) const
{
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());
	
	auto devicePointer = _cache->acquire(this);
	auto target        = _cache->acquire(m);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::equals(resultPointer, devicePointer, target, size());
	
	_cache->release(result);
	_cache->release(this);
	_cache->release(m);
	
	return result;
}

float CudaBlockSparseMatrix::reduceSum() const
{
	auto devicePointer = _cache->acquire(this);
	
	float result = CudaSparseMatrixLibrary::reduceSum(devicePointer, size());
	
	_cache->release(this);
	
	return result;
}

Value* CudaBlockSparseMatrix::reduceSumAlongColumns() const
{
	auto result = new CudaBlockSparseMatrix(*this, false);
	
	auto devicePointer = _cache->acquire(this);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::reduceSumAlongColumns(resultPointer, devicePointer,
		blocks(), rowsPerBlock(), columnsPerBlock(), isRowSparse());
	
	_cache->release(result);
	_cache->release(this);
	
	return result;
}

Value* CudaBlockSparseMatrix::reduceSumAlongRows() const
{
	auto result = new CudaBlockSparseMatrix(*this, false);
	
	auto devicePointer = _cache->acquire(this);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::reduceSumAlongRows(resultPointer, devicePointer,
		blocks(), rowsPerBlock(), columnsPerBlock(), isRowSparse());
	
	_cache->release(result);
	_cache->release(this);
	
	return result;
}

CudaBlockSparseMatrix::iterator CudaBlockSparseMatrix::begin()
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::begin();
}

CudaBlockSparseMatrix::const_iterator CudaBlockSparseMatrix::begin() const
{
	_cache->synchronize(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::begin();
}

CudaBlockSparseMatrix::iterator CudaBlockSparseMatrix::end()
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::end();
}

CudaBlockSparseMatrix::const_iterator CudaBlockSparseMatrix::end() const
{
	_cache->synchronize(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::end();
}

Matrix& CudaBlockSparseMatrix::front()
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::front();
}

const Matrix& CudaBlockSparseMatrix::front() const
{
	_cache->synchronize(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::front();
}

Matrix& CudaBlockSparseMatrix::back()
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::back();
}

const Matrix& CudaBlockSparseMatrix::back() const
{
	_cache->synchronize(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::back();
}

const Matrix& CudaBlockSparseMatrix::operator[](size_t position) const
{
	_cache->synchronize(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::operator[](position);
}

Matrix& CudaBlockSparseMatrix::operator[](size_t position)
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::operator[](position);
}

void CudaBlockSparseMatrix::pop_back()
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::pop_back();
}

void CudaBlockSparseMatrix::push_back(const Matrix& m)
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::push_back(m);
}

void CudaBlockSparseMatrix::resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock)
{
	_cache->invalidate(this);
	
	return BlockSparseMatrixImplementation::resize(blocks, rowsPerBlock, columnsPerBlock);
}

void CudaBlockSparseMatrix::resize(size_t blocks)
{
	_cache->invalidate(this);
	
	return BlockSparseMatrixImplementation::resize(blocks);
}

MatrixVector& CudaBlockSparseMatrix::data()
{
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::data();
}

const MatrixVector& CudaBlockSparseMatrix::data() const
{
	_cache->synchronize(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::data();
}
	
MatrixVector& CudaBlockSparseMatrix::rawData()
{
	return BlockSparseMatrixImplementation::data();
}

Value* CudaBlockSparseMatrix::clone() const
{
	return new CudaBlockSparseMatrix(*this);
}

bool CudaBlockSparseMatrix::isSupported()
{
	if(!util::KnobDatabase::getKnobValue("CudaBlockSparseMatrix::Enable", true))
	{
		return false;
	}
	
	CudaSparseMatrixLibrary::load();

	return CudaSparseMatrixLibrary::loaded();
}

}

}



