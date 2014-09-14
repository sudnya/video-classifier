/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the GeneralDifferentiableSolver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>

#include <minerva/matrix/interface/BlockSparseMatrixVector.h>
#include <minerva/matrix/interface/BlockSparseMatrix.h>
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace optimizer
{

typedef GeneralDifferentiableSolver::BlockSparseMatrixVector BlockSparseMatrixVector;
typedef GeneralDifferentiableSolver::BlockSparseMatrix       BlockSparseMatrix;
typedef GeneralDifferentiableSolver::Matrix                  Matrix;

GeneralDifferentiableSolver::~GeneralDifferentiableSolver()
{

}

float GeneralDifferentiableSolver::solve(Matrix& inputs, const CostAndGradientFunction& callBack)
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

float GeneralDifferentiableSolver::solve(BlockSparseMatrix& inputs, const CostAndGradientFunction& callBack)
{
	BlockSparseMatrixVector blockSparseInputs;

	blockSparseInputs.reserve(1);
	
	blockSparseInputs.push_back(std::move(inputs));
	
	float result = solve(blockSparseInputs, callBack);
	
	inputs = std::move(blockSparseInputs[0]);
	
	return result;
}


}

}

