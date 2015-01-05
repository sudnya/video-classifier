/*! \file   BlockSparseMatrixImplementation.cpp
	\author Gregory Diamos
	\date   Sunday December 29, 2013
	\brief  The source file for the BlockSparseMatrixImplementatio class.
*/

// Minerva Includes
#include <minerva/matrix/interface/BlockSparseMatrixImplementation.h>
#include <minerva/matrix/interface/NaiveBlockSparseMatrix.h>
#include <minerva/matrix/interface/CudaBlockSparseMatrix.h>
#include <minerva/matrix/interface/Matrix.h>

// Standard Library Includes
#include <sstream>

namespace minerva
{

namespace matrix
{

typedef BlockSparseMatrixImplementation::Value Value;
typedef BlockSparseMatrixImplementation::MatrixVector MatrixVector;

BlockSparseMatrixImplementation::BlockSparseMatrixImplementation(size_t blocks, size_t rows,
	size_t columns, bool rowSparse)
: _isRowSparse(rowSparse)
{
	resize(blocks, rows, columns);
}

BlockSparseMatrixImplementation::~BlockSparseMatrixImplementation()
{

}

BlockSparseMatrixImplementation::iterator BlockSparseMatrixImplementation::begin()
{
	return _matrices.begin();
}

BlockSparseMatrixImplementation::const_iterator BlockSparseMatrixImplementation::begin() const
{
	return _matrices.begin();
}

BlockSparseMatrixImplementation::iterator BlockSparseMatrixImplementation::end()
{
	return _matrices.end();
}

BlockSparseMatrixImplementation::const_iterator BlockSparseMatrixImplementation::end() const
{
	return _matrices.end();
}

Matrix& BlockSparseMatrixImplementation::front()
{
	return _matrices.front();
}

const Matrix& BlockSparseMatrixImplementation::front() const
{
	return _matrices.front();
}

Matrix& BlockSparseMatrixImplementation::back()
{
	return _matrices.back();
}

const Matrix& BlockSparseMatrixImplementation::back() const
{
	return _matrices.back();
}

const Matrix& BlockSparseMatrixImplementation::operator[](size_t position) const
{
	return _matrices[position];
}

Matrix& BlockSparseMatrixImplementation::operator[](size_t position)
{
	return _matrices[position];
}

BlockSparseMatrixImplementation::FloatReference BlockSparseMatrixImplementation::operator()(size_t row, size_t column)
{
	size_t block = isRowSparse() ? row / rowsPerBlock() : column / columnsPerBlock();
	
	size_t blockRow    = isRowSparse()    ? row    % rowsPerBlock()    : row;
	size_t blockColumn = isColumnSparse() ? column % columnsPerBlock() : column;
	
	return (*this)[block](blockRow, blockColumn);
}

BlockSparseMatrixImplementation::ConstFloatReference BlockSparseMatrixImplementation::operator()(size_t row, size_t column) const
{
	size_t block = isRowSparse() ? row / rowsPerBlock() : column / columnsPerBlock();
	
	size_t blockRow    = isRowSparse()    ? row    % rowsPerBlock()    : row;
	size_t blockColumn = isColumnSparse() ? column % columnsPerBlock() : column;
	
	return (*this)[block](blockRow, blockColumn);
}

void BlockSparseMatrixImplementation::pop_back()
{
	return _matrices.pop_back();
}

void BlockSparseMatrixImplementation::push_back(const Matrix& m)
{
	return _matrices.push_back(m);
}

size_t BlockSparseMatrixImplementation::size() const
{
	if(blocks() == 0) return 0;

	return blocks() * rowsPerBlock() * columnsPerBlock();
}

size_t BlockSparseMatrixImplementation::blocks() const
{
	return _matrices.size();
}

bool BlockSparseMatrixImplementation::empty() const
{
	return _matrices.empty();
}
	
size_t BlockSparseMatrixImplementation::getBlockingFactor() const
{
	if(isRowSparse())
	{
		return rowsPerBlock();
	}
	else
	{
		return columnsPerBlock();
	}
}

size_t BlockSparseMatrixImplementation::columns() const
{
	if(empty()) return 0;

	if(isColumnSparse())
	{
		size_t c = 0;

		for(auto& matrix : *this)
		{
			c += matrix.columns();
		}

		return c;
	}
	
	return front().columns();
}

size_t BlockSparseMatrixImplementation::rows() const
{
	if(empty()) return 0;
	
	if(isRowSparse())
	{
		size_t r = 0;

		for(auto& matrix : *this)
		{
			r += matrix.rows();
		}

		return r;
	}
	
	return front().rows();
}
    
size_t BlockSparseMatrixImplementation::columnsPerBlock() const
{
	if(empty()) return 0;
	
	return front().columns();
}

size_t BlockSparseMatrixImplementation::rowsPerBlock() const
{
	if(empty()) return 0;
	
	return front().rows();
}

MatrixVector& BlockSparseMatrixImplementation::data()
{
	return _matrices;
}

const MatrixVector& BlockSparseMatrixImplementation::data() const
{
	return _matrices;
}

std::string BlockSparseMatrixImplementation::toString() const
{
	if(empty()) return "(0 rows, 0 columns) []";

	std::stringstream stream;

	stream << "((" << blocks() << " blocks, " << rows()
		<< " rows, " << columns() << " columns)) - [ " << front().toString();

	return stream.str();
}

std::string BlockSparseMatrixImplementation::debugString() const
{
	return toString();
}

std::string BlockSparseMatrixImplementation::shapeString() const
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
	
void BlockSparseMatrixImplementation::resize(size_t blocks, size_t rowsPerBlock, size_t columnsPerBlock)
{
	_matrices.resize(blocks);
	
	for(auto& matrix : *this)
	{
		matrix.resize(rowsPerBlock, columnsPerBlock);
	}
}

Matrix BlockSparseMatrixImplementation::slice(size_t startRow, size_t startColumn, size_t rows, size_t columns) const
{
	// TODO
	return toMatrix().slice(startRow, startColumn, rows, columns);
}

void BlockSparseMatrixImplementation::assign(size_t startRow, size_t startColumn, const Matrix& m)
{
	// TODO: better
	for(size_t r = 0; r < m.rows(); ++r)
	{
		for(size_t c = 0; c < m.columns(); ++c)
		{
			(*this)(startRow + r, startColumn + c) = m(r, c);
		}
	}
}	

Matrix BlockSparseMatrixImplementation::toMatrix() const
{
	Matrix result(rows(), columns());

	if(isColumnSparse())
	{
		size_t column = 0;
		
		for(auto& matrix : *this)
		{
			// TODO: faster	
			
			size_t rows = matrix.rows();

			for(size_t row = 0; row < rows; ++row)
			{
				std::memcpy(&result.data()[result.getPosition(row, column)],
					&matrix.data()[matrix.getPosition(row, 0)],
					matrix.columns() * sizeof(float));
			}
			column += matrix.columns();
		}
	}
	else
	{
		size_t row = 0;

		for(auto& matrix : *this)
		{
			std::memcpy(&result.data()[result.getPosition(row, 0)],
				matrix.data().data(), matrix.size() * sizeof(float));
			row += matrix.rows();
		}
	}
	
	return result;
}

void BlockSparseMatrixImplementation::resize(size_t blocks)
{
	_matrices.resize(blocks);
}

bool& BlockSparseMatrixImplementation::isRowSparse()
{
	return _isRowSparse;
}

bool BlockSparseMatrixImplementation::isRowSparse() const
{
	return _isRowSparse;
}

bool BlockSparseMatrixImplementation::isColumnSparse() const
{
	return not isRowSparse();
}

Value* BlockSparseMatrixImplementation::createBestImplementation(size_t blocks, size_t rows,
	size_t columns, bool isRowSparse)
{
	Value* matrix = nullptr;
	
	if(matrix == nullptr && CudaBlockSparseMatrix::isSupported())
	{
		matrix = new CudaBlockSparseMatrix(blocks, rows, columns, isRowSparse);
	}
	
	if(matrix == nullptr)
	{	
		matrix = new NaiveBlockSparseMatrix(blocks, rows, columns, isRowSparse);
	}
	
	return matrix;
}

}

}


