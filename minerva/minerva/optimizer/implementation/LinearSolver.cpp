/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the LinearSolver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/LinearSolver.h>

#include <minerva/matrix/interface/BlockSparseMatrix.h>

namespace minerva
{

namespace optimizer
{

LinearSolver::~LinearSolver()
{

}

LinearSolver::CostAndGradient::CostAndGradient(float i, float c,
	const DataStructureFormat& f)
: initialCost(i), costReductionFactor(c), format(f)
{

}

LinearSolver::CostAndGradient::~CostAndGradient()
{

}

LinearSolver::SparseMatrixFormat::SparseMatrixFormat(size_t b, size_t r, size_t c)
: blocks(b), rowsPerBlock(r), columnsPerBlock(c)
{

}

LinearSolver::SparseMatrixFormat::SparseMatrixFormat(const BlockSparseMatrix& matrix)
: blocks(matrix.blocks()), rowsPerBlock(matrix.rowsPerBlock()), columnsPerBlock(matrix.columnsPerBlock())
{

}

BlockSparseMatrixVector LinearSolver::CostAndGradient::getUninitializedDataStructure() const
{
	BlockSparseMatrixVector vector;
	
	vector.reserve(sparseMatrixCount);
	
	for(auto& sparseMatrixFormat : format)
	{
		vector.push_back(BlockSparseMatrix(sparseMatrixFormat.blocks,
			sparseMatrixFormat.rowsPerBlock,
			sparseMatrixFormat.columnsPerBlock));
	}
	
	return vector;
}

}

}

