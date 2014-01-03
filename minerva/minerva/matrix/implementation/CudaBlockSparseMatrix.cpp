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
: BlockSparseMatrixImplementation(m.blocks(), m.rowsPerBlock(), m.columnsPerBlock(), m.isRowSparse()), _isTransposed(false)
{
	if(copyData)
	{
		auto copy = m.begin();
		
		assert(!m._isTransposed);
		
		for(auto matrix = _matrices.begin(); matrix != _matrices.end(); ++matrix, ++copy)
		{
			*matrix = *copy;
		}
	}
	else
	{
		_isTransposed = m._isTransposed;
	}
}

CudaBlockSparseMatrix::CudaBlockSparseMatrix(const CudaBlockSparseMatrix& m)
: BlockSparseMatrixImplementation(m.blocks(), m.rowsPerBlock(), m.columnsPerBlock(), m.isRowSparse()), _isTransposed(false)
{
	auto copy = m.begin();
	
	assert(!m._isTransposed);
	
	for(auto matrix = _matrices.begin(); matrix != _matrices.end(); ++matrix, ++copy)
	{
		*matrix = *copy;
	}
}

CudaBlockSparseMatrix::~CudaBlockSparseMatrix()
{
	_cache->invalidate(this);
}

Value* CudaBlockSparseMatrix::multiply(const Value* m) const
{
	// TODO: in parallel
	#if 0
	_performTransposeIfNecessary(m);
	
	assert(_isTransposed == static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed);

	auto result = new CudaBlockSparseMatrix(*this, false);
	
	result->resize(blocks());

	assert(m->blocks() == blocks());
	assertM(columns() == m->rows(), "Left columns " << columns() << " does not match right rows " << m->rows());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->multiply(*right));
	}
	
	return result;
	#else 
	assert(m->blocks() == blocks());

	auto matrixPointer = _cache->acquireReadOnly(m);
	auto devicePointer = _cache->acquireReadOnly(this);	

	size_t resultRows    = rowsPerBlock();
	size_t resultColumns = m->columnsPerBlock();

	auto result = new CudaBlockSparseMatrix(blocks(), resultRows, resultColumns, isRowSparse());
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::multiply(resultPointer, devicePointer, _isTransposed, matrixPointer,
		static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed,
		blocks(), rowsPerBlock(), columnsPerBlock(), m->rowsPerBlock(), m->columnsPerBlock());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);

	return result;
	#endif
}

Value* CudaBlockSparseMatrix::multiply(float f) const
{
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	
	assert(result->_isTransposed == _isTransposed);

	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::multiply(resultPointer, devicePointer, f, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
		
	return result;
}

Value* CudaBlockSparseMatrix::elementMultiply(const Value* m) const
{
	_performTransposeIfNecessary(m);
	assert(_isTransposed == static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed);
	
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());

	auto matrixPointer = _cache->acquireReadOnly(m);
	auto devicePointer = _cache->acquireReadOnly(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::elementMultiply(resultPointer, devicePointer, matrixPointer, size());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;
}

Value* CudaBlockSparseMatrix::add(const Value* m) const
{
	_performTransposeIfNecessary(m);
	
	assert(_isTransposed == static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed);
	
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());

	auto matrixPointer = _cache->acquireReadOnly(m);
	auto devicePointer = _cache->acquireReadOnly(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::add(resultPointer, devicePointer, matrixPointer, size());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;
}

Value* CudaBlockSparseMatrix::addBroadcastRow(const Value* m) const
{
	_performTransposeIfNecessary();
	_performTransposeIfNecessary(m);

	assert(!_isTransposed);
	assert(!static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed);

	auto result = new CudaBlockSparseMatrix(*this, false);

	auto matrixPointer = _cache->acquireReadOnly(m);
	auto devicePointer = _cache->acquireReadOnly(this);	
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
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::add(resultPointer, devicePointer, f, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
	
	return result;
}

Value* CudaBlockSparseMatrix::subtract(const Value* m) const
{
	_performTransposeIfNecessary(m);
	assert(_isTransposed == static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed);
	
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());
	
	auto matrixPointer = _cache->acquireReadOnly(m);
	auto devicePointer = _cache->acquireReadOnly(this);	
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::subtract(resultPointer, devicePointer, matrixPointer, size());

	_cache->release(this);
	_cache->release(result);
	_cache->release(m);
	
	return result;

}
Value* CudaBlockSparseMatrix::subtract(float f) const
{
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::subtract(resultPointer, devicePointer, f, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
	
	return result;
}

Value* CudaBlockSparseMatrix::log() const
{
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::log(resultPointer, devicePointer, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
	
	return result;
}

Value* CudaBlockSparseMatrix::negate() const
{
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::negate(resultPointer, devicePointer, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
	
	return result;
}

Value* CudaBlockSparseMatrix::sigmoid() const
{
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::sigmoid(resultPointer, devicePointer, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
	
	return result;
}

Value* CudaBlockSparseMatrix::sigmoidDerivative() const
{
	auto devicePointer = _cache->acquireReadOnly(this);	
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::sigmoidDerivative(resultPointer, devicePointer, size());

	_cache->release(this);
	_cache->release(result);
	
	assert(result->_isTransposed == _isTransposed);
	
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
	util::log("CudaBlockSparseMatrix") << "Transposing matrix " << this << ": " << _isTransposed << "\n";
	#if 0
	auto result = new CudaBlockSparseMatrix(*this);
	
	result->transposeSelf();
	
	return result;
	#else
	
	auto devicePointer = _cache->acquireReadOnly(this);	
	assert(!_isTransposed);
	util::log("CudaBlockSparseMatrix") << " pointer is " << devicePointer << "\n";
	
	auto result = new CudaBlockSparseMatrix(blocks(), columnsPerBlock(), rowsPerBlock(), isRowSparse());
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::transpose(resultPointer, devicePointer, blocks(), rowsPerBlock(), columnsPerBlock());
	
	//result->transposeSelf();
	//assert(result->_isTransposed != _isTransposed);

	_cache->release(this);
	_cache->release(result);

	//result->_performTransposeIfNecessary();
	
	return result;
	
	#endif
}

void CudaBlockSparseMatrix::negateSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::negateSelf(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::logSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::logSelf(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::sigmoidSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::sigmoidSelf(devicePointer, size());
	
	_cache->release(this);
}

void CudaBlockSparseMatrix::sigmoidDerivativeSelf()
{
	auto devicePointer = _cache->acquire(this);
	
	CudaSparseMatrixLibrary::sigmoidDerivativeSelf(devicePointer, size());
	
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
	
	//util::log("CudaBlockSparseMatrixLibrary") << " result is: " << toString();
	#else
	
	for(auto& matrix : *this)
	{
		matrix.assignUniformRandomValues(engine, min, max);
	}
	#endif
}

Value* CudaBlockSparseMatrix::greaterThanOrEqual(float f) const
{
	auto devicePointer = _cache->acquireReadOnly(this);
	
	auto result = new CudaBlockSparseMatrix(*this, false);
	auto resultPointer = _cache->acquireClobber(this);
	
	CudaSparseMatrixLibrary::greaterThanOrEqual(resultPointer, devicePointer, f, size());
	
	_cache->release(result);
	_cache->release(this);
	
	assert(result->_isTransposed == _isTransposed);
	
	return result;
}

Value* CudaBlockSparseMatrix::equals(const Value* m) const
{
	_performTransposeIfNecessary(m);
	assert(_isTransposed == static_cast<const CudaBlockSparseMatrix*>(m)->_isTransposed);
	
	auto result = new CudaBlockSparseMatrix(*this, false);

	assert(m->size() == size());
	
	auto devicePointer = _cache->acquireReadOnly(this);
	auto target        = _cache->acquireReadOnly(m);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::equals(resultPointer, devicePointer, target, size());
	
	_cache->release(result);
	_cache->release(this);
	_cache->release(m);
	
	return result;
}

float CudaBlockSparseMatrix::reduceSum() const
{
	auto devicePointer = _cache->acquireReadOnly(this);
	
	float result = CudaSparseMatrixLibrary::reduceSum(devicePointer, size());
	
	_cache->release(this);
	
	return result;
}

Value* CudaBlockSparseMatrix::reduceSumAlongColumns() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);

	auto result = new CudaBlockSparseMatrix(*this, false);
	
	auto devicePointer = _cache->acquireReadOnly(this);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::reduceSumAlongColumns(resultPointer, devicePointer,
		blocks(), rowsPerBlock(), columnsPerBlock(), isRowSparse());
	
	_cache->release(result);
	_cache->release(this);
	
	return result;
}

Value* CudaBlockSparseMatrix::reduceSumAlongRows() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	auto result = new CudaBlockSparseMatrix(*this, false);
	
	auto devicePointer = _cache->acquireReadOnly(this);
	auto resultPointer = _cache->acquireClobber(result);
	
	CudaSparseMatrixLibrary::reduceSumAlongRows(resultPointer, devicePointer,
		blocks(), rowsPerBlock(), columnsPerBlock(), isRowSparse());
	
	_cache->release(result);
	_cache->release(this);
	
	return result;
}

CudaBlockSparseMatrix::iterator CudaBlockSparseMatrix::begin()
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::begin();
}

CudaBlockSparseMatrix::const_iterator CudaBlockSparseMatrix::begin() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronizeHostReadOnly(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::begin();
}

CudaBlockSparseMatrix::iterator CudaBlockSparseMatrix::end()
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::end();
}

CudaBlockSparseMatrix::const_iterator CudaBlockSparseMatrix::end() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronizeHostReadOnly(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::end();
}

Matrix& CudaBlockSparseMatrix::front()
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::front();
}

const Matrix& CudaBlockSparseMatrix::front() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronizeHostReadOnly(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::front();
}

Matrix& CudaBlockSparseMatrix::back()
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::back();
}

const Matrix& CudaBlockSparseMatrix::back() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronizeHostReadOnly(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::back();
}

const Matrix& CudaBlockSparseMatrix::operator[](size_t position) const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronizeHostReadOnly(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::operator[](position);
}

Matrix& CudaBlockSparseMatrix::operator[](size_t position)
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::operator[](position);
}

void CudaBlockSparseMatrix::pop_back()
{
	_cache->invalidate(this);
	
	return BlockSparseMatrixImplementation::pop_back();
}

void CudaBlockSparseMatrix::push_back(const Matrix& m)
{
	_cache->invalidate(this);
	
	return BlockSparseMatrixImplementation::push_back(m);
}

size_t CudaBlockSparseMatrix::columns() const
{
	if(isRowSparse())
	{
		return columnsPerBlock();
	}
	else
	{
		return blocks() * columnsPerBlock();
	}
}

size_t CudaBlockSparseMatrix::rows() const
{
	if(isRowSparse())
	{
		return blocks() * rowsPerBlock();
	}
	else
	{
		return rowsPerBlock();
	}
}

size_t CudaBlockSparseMatrix::columnsPerBlock() const
{
	if(_isTransposed)
	{
		return _matrices.front().rows();
	}
	else
	{
		return _matrices.front().columns();
	}
}

size_t CudaBlockSparseMatrix::rowsPerBlock() const
{
	if(_isTransposed)
	{
		return _matrices.front().columns();
	}
	else
	{
		return _matrices.front().rows();
	}
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
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronize(this);
	
	return BlockSparseMatrixImplementation::data();
}

const MatrixVector& CudaBlockSparseMatrix::data() const
{
	_performTransposeIfNecessary();
	assert(!_isTransposed);
	_cache->synchronizeHostReadOnly(const_cast<CudaBlockSparseMatrix*>(this));
	
	return BlockSparseMatrixImplementation::data();
}
	
MatrixVector& CudaBlockSparseMatrix::rawData()
{
	_performTransposeIfNecessary();
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
	
void CudaBlockSparseMatrix::_performTransposeIfNecessary() const
{
	if(!_isTransposed) return;

	auto matrix = const_cast<CudaBlockSparseMatrix*>(this);
	
	matrix->_isTransposed = false;
	
	_cache->synchronize(this);
	util::log("CudaBlockSparseMatrix") << "Explicitly transposing matrix " << matrix << "\n";
	
	for(auto& block : matrix->rawData())
	{
		block.transposeSelf();
	}
}

void CudaBlockSparseMatrix::_performTransposeIfNecessary(const BlockSparseMatrixImplementation* matrix) const
{
	auto m = static_cast<const CudaBlockSparseMatrix*>(matrix);

	if(_isTransposed == m->_isTransposed) return;

	_performTransposeIfNecessary();
	m->_performTransposeIfNecessary();
}

}

}



