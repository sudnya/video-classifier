/*	\file   BlockSparseMatrix.cpp
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the BlockSparseMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/BlockSparseMatrixImplementation.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace matrix
{

BlockSparseMatrix::BlockSparseMatrix(size_t blocks, size_t rows,
	size_t columns, bool isRowSparse)
: _implementation(BlockSparseMatrixImplementation::createBestImplementation(blocks, rows, columns, isRowSparse))
{

}

BlockSparseMatrix::BlockSparseMatrix(bool isRowSparse)
: _implementation(BlockSparseMatrixImplementation::createBestImplementation(0, 0, 0, isRowSparse))
{

}

BlockSparseMatrix::BlockSparseMatrix(const BlockSparseMatrix& m)
: _implementation(nullptr)
{
	if(m._implementation != nullptr)
	{
		_implementation = m._implementation->clone();
	}
}

BlockSparseMatrix::BlockSparseMatrix(BlockSparseMatrix&& m)
: _implementation(nullptr)
{
	std::swap(_implementation, m._implementation);
}

BlockSparseMatrix::~BlockSparseMatrix()
{
	delete _implementation;
}
	
BlockSparseMatrix& BlockSparseMatrix::operator=(const BlockSparseMatrix& m)
{
	if(this == &m) return *this;
	
	delete _implementation;
	
	if(m._implementation != nullptr)
	{
		_implementation = m._implementation->clone();
	}
	
	return *this;
}

BlockSparseMatrix& BlockSparseMatrix::operator=(BlockSparseMatrix&& m)
{
	std::swap(_implementation, m._implementation);
	
	return *this;
}

BlockSparseMatrix::iterator BlockSparseMatrix::begin()
{
	return _implementation->begin();
}

BlockSparseMatrix::const_iterator BlockSparseMatrix::begin() const
{
	return _implementation->begin();
}

BlockSparseMatrix::iterator BlockSparseMatrix::end()
{
	return _implementation->end();
}

BlockSparseMatrix::const_iterator BlockSparseMatrix::end() const
{
	return _implementation->end();
}

Matrix& BlockSparseMatrix::front()
{
	return _implementation->front();
}

const Matrix& BlockSparseMatrix::front() const
{
	return _implementation->front();
}

Matrix& BlockSparseMatrix::back()
{
	return _implementation->back();
}

const Matrix& BlockSparseMatrix::back() const
{
	return _implementation->back();
}

const Matrix& BlockSparseMatrix::operator[](size_t position) const
{
	return (*_implementation)[position];
}

Matrix& BlockSparseMatrix::operator[](size_t position)
{
	return (*_implementation)[position];
}

void BlockSparseMatrix::pop_back()
{
	return _implementation->pop_back();
}

void BlockSparseMatrix::push_back(const Matrix& m)
{
	return _implementation->push_back(m);
}

size_t BlockSparseMatrix::size() const
{
	return _implementation->size();
}

size_t BlockSparseMatrix::blocks() const
{
	return _implementation->blocks();
}

bool BlockSparseMatrix::empty() const
{
	return _implementation->empty();
}

size_t BlockSparseMatrix::columns() const
{
	return _implementation->columns();
}

size_t BlockSparseMatrix::rows() const
{
	return _implementation->rows();
}
	
size_t BlockSparseMatrix::rowsPerBlock() const
{
	return _implementation->rowsPerBlock();
}

size_t BlockSparseMatrix::columnsPerBlock() const
{
	return _implementation->columnsPerBlock();
}

bool BlockSparseMatrix::isRowSparse() const
{
	return _implementation->isRowSparse();
}

bool BlockSparseMatrix::isColumnSparse() const
{
	return not _implementation->isRowSparse();
}

size_t BlockSparseMatrix::getBlockingFactor() const
{
	if(blocks() == 0) return 0;
	
	if(isRowSparse()) return rowsPerBlock();

	return columnsPerBlock();
}

size_t BlockSparseMatrix::blockSize() const
{
	return rowsPerBlock() * columnsPerBlock();
}

void BlockSparseMatrix::resize(size_t blocks, size_t rowsPerBlock,
	size_t columnsPerBlock)
{
	_implementation->resize(blocks, rowsPerBlock, columnsPerBlock);
}

void BlockSparseMatrix::resize(size_t blocks)
{
	_implementation->resize(blocks);
}

void BlockSparseMatrix::setColumnSparse()
{
	_implementation->isRowSparse() = false;
}

void BlockSparseMatrix::setRowSparse()
{
	_implementation->isRowSparse() = true;
}

BlockSparseMatrix BlockSparseMatrix::multiply(
	const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->multiply(m._implementation));
}

BlockSparseMatrix BlockSparseMatrix::convolutionalMultiply(
	const BlockSparseMatrix& m, size_t step) const
{
	return BlockSparseMatrix(_implementation->convolutionalMultiply(
		m._implementation, step));
}

BlockSparseMatrix BlockSparseMatrix::reverseConvolutionalMultiply(
	const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->reverseConvolutionalMultiply(
		m._implementation));
}

BlockSparseMatrix BlockSparseMatrix::multiply(float f) const
{
	return BlockSparseMatrix(_implementation->multiply(f));
}

BlockSparseMatrix BlockSparseMatrix::elementMultiply(const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->elementMultiply(m._implementation));
}

BlockSparseMatrix BlockSparseMatrix::add(const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->add(m._implementation));
}

BlockSparseMatrix BlockSparseMatrix::addBroadcastRow(const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->addBroadcastRow(m._implementation));
}

BlockSparseMatrix BlockSparseMatrix::convolutionalAddBroadcastRow(
	const BlockSparseMatrix& m, size_t step) const
{
	return BlockSparseMatrix(_implementation->convolutionalAddBroadcastRow(
		m._implementation, step));
}

BlockSparseMatrix BlockSparseMatrix::add(float f) const
{
	return BlockSparseMatrix(_implementation->add(f));
}

BlockSparseMatrix BlockSparseMatrix::subtract(const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->subtract(m._implementation));
}

BlockSparseMatrix BlockSparseMatrix::subtract(float f) const
{
	return BlockSparseMatrix(_implementation->subtract(f));
}

BlockSparseMatrix BlockSparseMatrix::log() const
{
	return BlockSparseMatrix(_implementation->log());
}

BlockSparseMatrix BlockSparseMatrix::negate() const
{
	return BlockSparseMatrix(_implementation->negate());
}

BlockSparseMatrix BlockSparseMatrix::sigmoidDerivative() const
{
	return BlockSparseMatrix(_implementation->sigmoidDerivative());
}

BlockSparseMatrix BlockSparseMatrix::sigmoid() const
{
	return BlockSparseMatrix(_implementation->sigmoid());
}
	
BlockSparseMatrix BlockSparseMatrix::klDivergence(float sparsity) const
{
	return BlockSparseMatrix(_implementation->klDivergence(sparsity));
}

BlockSparseMatrix BlockSparseMatrix::klDivergenceDerivative(float sparsity) const
{
	return BlockSparseMatrix(_implementation->klDivergenceDerivative(sparsity));
}

BlockSparseMatrix BlockSparseMatrix::transpose() const
{
	return BlockSparseMatrix(_implementation->transpose());
}

void BlockSparseMatrix::negateSelf()
{
	_implementation->negateSelf();
}

void BlockSparseMatrix::logSelf()
{
	_implementation->logSelf();
}

void BlockSparseMatrix::sigmoidSelf()
{
	_implementation->sigmoidSelf();
}

void BlockSparseMatrix::sigmoidDerivativeSelf()
{
	_implementation->sigmoidDerivativeSelf();
}

void BlockSparseMatrix::transposeSelf()
{
	_implementation->transposeSelf();
}

void BlockSparseMatrix::assignUniformRandomValues(
	std::default_random_engine& engine, float min, float max)
{
	_implementation->assignUniformRandomValues(engine, min, max);
}

BlockSparseMatrix BlockSparseMatrix::greaterThanOrEqual(float f) const
{
	return BlockSparseMatrix(_implementation->greaterThanOrEqual(f));
}

BlockSparseMatrix BlockSparseMatrix::equals(const BlockSparseMatrix& m) const
{
	return BlockSparseMatrix(_implementation->equals(m._implementation));
}

Matrix BlockSparseMatrix::toMatrix() const
{
	return _implementation->toMatrix();
}

float BlockSparseMatrix::reduceSum() const
{
	return _implementation->reduceSum();
}

BlockSparseMatrix BlockSparseMatrix::reduceSumAlongColumns() const
{
	return BlockSparseMatrix(_implementation->reduceSumAlongColumns());
}

BlockSparseMatrix BlockSparseMatrix::reduceSumAlongRows() const
{
	return BlockSparseMatrix(_implementation->reduceSumAlongRows());
}

BlockSparseMatrix BlockSparseMatrix::reduceTileSumAlongRows(size_t tilesPerRow, size_t rowsPerBlock) const
{
	return BlockSparseMatrix(_implementation->reduceTileSumAlongRows(tilesPerRow, rowsPerBlock));
}

std::string BlockSparseMatrix::toString() const
{
	return _implementation->toString();
}

std::string BlockSparseMatrix::debugString() const
{
	return toString();
}

std::string BlockSparseMatrix::shapeString() const
{
	std::stringstream stream;

	stream << "(" << blocks() << " blocks, " << rowsPerBlock() << " rows per block, "
		<< columnsPerBlock() << " columns per block, ";
	
	if(isRowSparse())
	{
		stream << "row-sparse";
	}
	else
	{
		stream << "column-sparse";
	}
	
	stream << ")";

	return stream.str();
}

BlockSparseMatrix::BlockSparseMatrix(BlockSparseMatrixImplementation* i)
: _implementation(i)
{

}

}

}

