/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the LinearSolver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/LinearSolver.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace optimizer
{

typedef LinearSolver::DataStructureFormat     DataStructureFormat;
typedef SparseMatrixFormat                    SparseMatrixFormat;
typedef LinearSolver::BlockSparseMatrixVector BlockSparseMatrixVector;
typedef LinearSolver::BlockSparseMatrix       BlockSparseMatrix;
typedef LinearSolver::Matrix                  Matrix;

LinearSolver::~LinearSolver()
{

}

float LinearSolver::solve(Matrix& inputs, const CostAndGradient& callBack)
{
	BlockSparseMatrixVector blockSparseInputs;

	blockSparseInputs.reserve(1);
	
	blockSparseInputs.push_back(
			BlockSparseMatrix(1, inputs.rows(), inputs.columns()));
	
	blockSparseInputs[0][0] = std::move(inputs);
	
	float result = solve(blockSparseInputs, callBack);
	
	inputs = std::move(blockSparseInputs[0][0]);
	
	return result;
}

float LinearSolver::solve(BlockSparseMatrix& inputs, const CostAndGradient& callBack)
{
	BlockSparseMatrixVector blockSparseInputs;

	blockSparseInputs.reserve(1);
	
	blockSparseInputs.push_back(std::move(inputs));
	
	float result = solve(blockSparseInputs, callBack);
	
	inputs = std::move(blockSparseInputs[0]);
	
	return result;
}

LinearSolver::CostAndGradient::CostAndGradient(float i, float c,
	const DataStructureFormat& f)
: initialCost(i), costReductionFactor(c), format(f)
{

}

static DataStructureFormat convertToFormat(const BlockSparseMatrixVector& vector)
{
	DataStructureFormat format;
	
	for(auto& matrix : vector)
	{
		format.push_back(SparseMatrixFormat(matrix));
	}
	
	return format;
}

static DataStructureFormat convertToFormat(const Matrix& matrix)
{
	DataStructureFormat format;
	
	format.push_back(SparseMatrixFormat(matrix));
	
	return format;
}

LinearSolver::CostAndGradient::CostAndGradient(float i, float c,
	const BlockSparseMatrixVector& vector)
: initialCost(i), costReductionFactor(c), format(convertToFormat(vector))
{

}

LinearSolver::CostAndGradient::CostAndGradient(float i, float c,
	const Matrix& matrix)
: initialCost(i), costReductionFactor(c), format(convertToFormat(matrix))
{

}

LinearSolver::CostAndGradient::~CostAndGradient()
{

}

SparseMatrixFormat::SparseMatrixFormat(size_t b, size_t r, size_t c, bool s)
: blocks(b), rowsPerBlock(r), columnsPerBlock(c), isRowSparse(s)
{

}

SparseMatrixFormat::SparseMatrixFormat(const BlockSparseMatrix& matrix)
: blocks(matrix.blocks()), rowsPerBlock(matrix.rowsPerBlock()),
  columnsPerBlock(matrix.columnsPerBlock()), isRowSparse(matrix.isRowSparse())
{

}

SparseMatrixFormat::SparseMatrixFormat(const Matrix& matrix)
: blocks(1), rowsPerBlock(matrix.rows()), columnsPerBlock(matrix.columns()), isRowSparse(true)
{

}

BlockSparseMatrixVector LinearSolver::CostAndGradient::getUninitializedDataStructure() const
{
	BlockSparseMatrixVector vector;
	
	vector.reserve(format.size());
	
	for(auto& sparseMatrixFormat : format)
	{
		vector.push_back(BlockSparseMatrix(sparseMatrixFormat.blocks,
			sparseMatrixFormat.rowsPerBlock,
			sparseMatrixFormat.columnsPerBlock,
			sparseMatrixFormat.isRowSparse));
	}
	
	return vector;
}

}

}

