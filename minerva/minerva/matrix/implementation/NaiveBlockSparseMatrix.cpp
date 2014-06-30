/*! \file   NaiveBlockSparseMatrix.cpp
	\author Gregory Diamos
	\date   Sunday December 29, 2013
	\brief  The source file for the NaiveBlockSparseMatrix class.
*/

// Minerva Includes
#include <minerva/matrix/interface/NaiveBlockSparseMatrix.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <cassert>

namespace minerva
{

namespace matrix
{

typedef NaiveBlockSparseMatrix::Value Value;

NaiveBlockSparseMatrix::NaiveBlockSparseMatrix(size_t blocks, size_t rows,
	size_t columns, bool rowSparse)
: BlockSparseMatrixImplementation(blocks, rows, columns, rowSparse)
{

}

NaiveBlockSparseMatrix::NaiveBlockSparseMatrix(bool rowSparse)
: BlockSparseMatrixImplementation(0, 0, 0, rowSparse)
{

}

Value* NaiveBlockSparseMatrix::multiply(const Value* matrix) const
{
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);

	// TODO: in parallel
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());

	assert(m->blocks() == blocks());
	assertM(columns() == m->rows(), "Left columns " << columns()
		<< " does not match right rows " << m->rows());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end();
		++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->multiply(*right));
	}

	return result;
}

static void zeroExtendColumns(Matrix& m, size_t columns)
{
	if(m.columns() < columns)
	{
		m = m.appendColumns(Matrix(m.rows(), columns - m.columns()));
	}
}

Value* NaiveBlockSparseMatrix::convolutionalMultiply(const Value* matrix,
	size_t step) const
{
	// Just multiply if there is a 1 to 1 match between blocks
	if(matrix->rowsPerBlock() == step && matrix->blocks() == blocks())
	{
		return multiply(matrix);
	}

	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	// TODO: in parallel
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	auto flattened = toMatrix();
	
	for(size_t leftBegin = 0; leftBegin < flattened.columns(); leftBegin += step)
	{
		size_t extent = std::min(flattened.columns(), leftBegin + m->rowsPerBlock());
		
		auto temp = flattened.slice(0, leftBegin, flattened.rows(), extent - leftBegin);
		
		zeroExtendColumns(temp, m->rowsPerBlock());
	
		// TODO:	
		size_t rightBlockIndex = leftBegin * m->blocks() / flattened.columns();//(leftBegin * flattened.columns()) / (m->rowsPerBlock() * m->rows());
		assert(rightBlockIndex < m->blocks());		

		result->push_back(temp.multiply((*m)[rightBlockIndex]));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::reverseConvolutionalMultiply(const Value* matrix) const
{
	// Just multiply if there is a 1 to 1 match between blocks
	if(columnsPerBlock() == matrix->rowsPerBlock() && matrix->blocks() == blocks())
	{
		return multiply(matrix);
	}

	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	// TODO: in parallel
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	size_t leftColumnStep  = m->rowsPerBlock();
	size_t leftColumn      = 0;

	auto flattened = toMatrix();

	result->resize(m->blocks());

	auto resultBlock = result->begin();

	for(auto& right : *m)
	{
		Matrix temp(flattened.rows(), right.columns());
		
		for(size_t leftBegin = leftColumn; leftBegin < flattened.columns(); leftBegin += right.rows())
		{
			size_t leftColumnEnd = std::min(flattened.columns(), leftBegin + leftColumnStep);
		
			auto leftSlice = flattened.slice(0, leftBegin, flattened.rows(), leftColumnEnd - leftBegin);
			
			zeroExtendColumns(leftSlice, m->rowsPerBlock());
			
			temp = temp.add(leftSlice.multiply(right));
		}
		
		*resultBlock = std::move(temp);
		
		leftColumn += leftColumnStep;

		++resultBlock;
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::multiply(float f) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->multiply(f));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::elementMultiply(const Value* matrix) const
{
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	auto result = new NaiveBlockSparseMatrix(isRowSparse());
	
	result->resize(blocks());
	
	// TODO: in parallel
	assert(m->blocks() == blocks());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->elementMultiply(*right));
	}

	return result;
}

Value* NaiveBlockSparseMatrix::add(const Value* matrix) const
{
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	auto result = new NaiveBlockSparseMatrix(isRowSparse());
	
	result->resize(blocks());
	
	// TODO: in parallel
	assert(m->blocks() == blocks());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->add(*right));
	}

	return result;
}

Value* NaiveBlockSparseMatrix::addBroadcastRow(const Value* matrix) const
{
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	auto result = new NaiveBlockSparseMatrix(isRowSparse());
	
	result->resize(blocks());
	
	// TODO: in parallel
	assert(m->blocks() == blocks());
	assert(m->isRowSparse() == isRowSparse());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->addBroadcastRow(*right));
	}

	return result;
}

Value* NaiveBlockSparseMatrix::convolutionalAddBroadcastRow(const Value* matrix) const
{
	// Just add if there is a 1 to 1 match between blocks
	if(blocks() == matrix->blocks())
	{
		return addBroadcastRow(matrix);
	}
	
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	auto result = new NaiveBlockSparseMatrix(isRowSparse());
	
	result->resize(blocks());
	
	assert(!isRowSparse());
	assert(m->isRowSparse() == isRowSparse());

	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto left = begin(); left != end(); ++left, ++resultBlock)
	{
		size_t rightIndex = (std::distance(begin(), left) * m->blocks()) / blocks();
		
		*resultBlock = std::move(left->addBroadcastRow((*m)[rightIndex]));
	}

	return result;
}

Value* NaiveBlockSparseMatrix::add(float f) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->add(f));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::subtract(const Value* matrix) const
{
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	auto result = new NaiveBlockSparseMatrix(isRowSparse());
	
	result->resize(blocks());
	
	// TODO: in parallel
	assert(m->blocks() == blocks());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->subtract(*right));
	}

	return result;
	
}

Value* NaiveBlockSparseMatrix::subtract(float f) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->subtract(f));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::log() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->log());
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::negate() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->negate());
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::sigmoid() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->sigmoid());
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::sigmoidDerivative() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->sigmoidDerivative());
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::klDivergence(float sparsity) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->klDivergence(sparsity));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::klDivergenceDerivative(float sparsity) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->klDivergenceDerivative(sparsity));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::transpose() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->transpose());
	}
	
	return result;
}

void NaiveBlockSparseMatrix::negateSelf()
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.negateSelf();
	}
}

void NaiveBlockSparseMatrix::logSelf()
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.logSelf();
	}
}

void NaiveBlockSparseMatrix::sigmoidSelf()
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.sigmoidSelf();
	}
}

void NaiveBlockSparseMatrix::sigmoidDerivativeSelf()
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.sigmoidDerivativeSelf();
	}
}

void NaiveBlockSparseMatrix::minSelf(float v)
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.minSelf(v);
	}
}

void NaiveBlockSparseMatrix::maxSelf(float v)
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.maxSelf(v);
	}
}

void NaiveBlockSparseMatrix::assignSelf(float v)
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.assignSelf(v);
	}
}

void NaiveBlockSparseMatrix::transposeSelf()
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.transposeSelf();
	}
}

void NaiveBlockSparseMatrix::assignUniformRandomValues(
	std::default_random_engine& engine, float min, float max)
{
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		matrix.assignUniformRandomValues(engine, min, max);
	}
}

Value* NaiveBlockSparseMatrix::greaterThanOrEqual(float f) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	result->resize(blocks());
	
	// TODO: in parallel
	auto resultBlock = result->begin();
	for(auto matrix = begin(); matrix != end(); ++matrix, ++resultBlock)
	{
		*resultBlock = std::move(matrix->greaterThanOrEqual(f));
	}
	
	return result;
}

Value* NaiveBlockSparseMatrix::equals(const Value* matrix) const
{
	auto m = dynamic_cast<const NaiveBlockSparseMatrix*>(matrix);
	assert(m != nullptr);
	
	auto result = new NaiveBlockSparseMatrix(isRowSparse());
	
	result->resize(blocks());
	
	// TODO: in parallel
	assert(m->blocks() == blocks());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->equals(*right));
	}

	return result;
}

float NaiveBlockSparseMatrix::reduceSum() const
{
	float sum = 0.0f;
	
	// TODO: in parallel
	for(auto& matrix : *this)
	{
		sum += matrix.reduceSum();
	}
	
	return sum;
}

Value* NaiveBlockSparseMatrix::reduceSumAlongColumns() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	// TODO: in parallel
	if(isColumnSparse())
	{
		if(!empty())
		{
			auto matrix = begin();
			
			auto resultMatrix = matrix->reduceSumAlongColumns();

			for(++matrix; matrix != end(); ++matrix)
			{
				resultMatrix = resultMatrix.add(matrix->reduceSumAlongColumns());
			}
			
			result->push_back(resultMatrix);
		}
	}
	else
	{
		result->resize(blocks());

		auto resultBlock = result->begin();		
		for(auto& matrix : *this)
		{
			*resultBlock = std::move(matrix.reduceSumAlongColumns());

			++resultBlock;
		}
	}
	
	return result;

}

Value* NaiveBlockSparseMatrix::reduceSumAlongRows() const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	// TODO: in parallel
	if(isRowSparse())
	{
		if(!empty())
		{
			auto matrix = begin();
			
			auto resultMatrix = matrix->reduceSumAlongRows();

			for(++matrix; matrix != end(); ++matrix)
			{
				resultMatrix = resultMatrix.add(matrix->reduceSumAlongRows());
			}
			
			result->push_back(resultMatrix);
		}
	}
	else
	{
		result->resize(blocks());

		auto resultBlock = result->begin();		
		for(auto& matrix : *this)
		{
			*resultBlock = std::move(matrix.reduceSumAlongRows());

			++resultBlock;
		}
	}
	
	return result;

}

static Matrix extractBlockByRow(const NaiveBlockSparseMatrix& matrix,
	size_t block, size_t row, size_t rows)
{
	Matrix result;
	
	size_t remainingRows = std::min(rows, matrix.rowsPerBlock() - row);
	
	result = matrix[block].slice(row, 0, remainingRows, matrix.columnsPerBlock());
	
	if(remainingRows < rows)
	{
		remainingRows = rows - remainingRows;
		block = block + 1;
		row   = 0;

		assert(remainingRows < matrix.rowsPerBlock());
		
		result = result.appendRows(matrix[block].slice(row, 0, remainingRows,
			matrix.columnsPerBlock()));
	}

	return result;
}

Value* NaiveBlockSparseMatrix::reduceTileSumAlongRows(size_t rowsPerTile,
	size_t blocks) const
{
	auto result = new NaiveBlockSparseMatrix(isRowSparse());

	size_t rowsPerStep = rowsPerTile * blocks;
	
	if(isRowSparse())
	{
		// step one tile at a time
		for(size_t row = 0; row < rows(); row += rowsPerStep)
		{
			size_t endingRow = row + rowsPerStep;
			
			assert(endingRow <= rows());
			
			if(result->empty())
			{
				for(size_t currentRow = row; currentRow < endingRow;
					currentRow += rowsPerTile)
				{
					size_t block       = currentRow / this->rowsPerBlock();
					size_t blockOffset = currentRow % this->rowsPerBlock();

					assert(block < this->blocks());

					if(blockOffset == 0 && this->rowsPerBlock() == rowsPerTile)
					{
						result->push_back((*this)[block]);
					}
					else
					{
						size_t rows = std::min(rowsPerTile, endingRow - currentRow);

						result->push_back((*this)[block].slice(
							0, 0, rows, columnsPerBlock()));
					}
				}
			}
			else
			{
				for(size_t currentRow = row; currentRow < endingRow;
					currentRow += rowsPerTile)
				{
					size_t block       = currentRow / this->rowsPerBlock();
					size_t blockOffset = currentRow % this->rowsPerBlock();

					size_t resultBlock =
						(currentRow / rowsPerBlock()) % result->blocks();
					
					assert(resultBlock < result->blocks());
					assert(block < this->blocks());

					if(blockOffset == 0 && this->rowsPerBlock() == rowsPerTile)
					{
						(*result)[resultBlock] = (*result)[resultBlock].add(
							(*this)[block]);
					}
					else
					{
						size_t rows = std::min(rowsPerTile, endingRow - currentRow);

						(*result)[resultBlock] = (*result)[resultBlock].add(
							extractBlockByRow(*this, block, blockOffset, rows));
					}
				}
			}
		}
	}
	else
	{
		assertM(false, "Not implemented.");
	}

	return result;
}

Value* NaiveBlockSparseMatrix::clone() const
{
	return new NaiveBlockSparseMatrix(*this);
}

}

}


