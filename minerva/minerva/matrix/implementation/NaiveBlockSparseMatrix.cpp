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
	assertM(columns() == m->rows(), "Left columns " << columns() << " does not match right rows " << m->rows());

	auto resultBlock = result->begin();
	for(auto left = begin(), right = m->begin(); left != end(); ++left, ++right, ++resultBlock)
	{
		*resultBlock = std::move(left->multiply(*right));
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

Value* NaiveBlockSparseMatrix::clone() const
{
	return new NaiveBlockSparseMatrix(*this);
}

}

}


