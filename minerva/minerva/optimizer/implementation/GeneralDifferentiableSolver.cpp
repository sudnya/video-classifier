/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the GeneralDifferentiableSolver class 
 */

// Minerva Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>

#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
{

namespace optimizer
{

typedef GeneralDifferentiableSolver::MatrixVector MatrixVector;
typedef GeneralDifferentiableSolver::Matrix       Matrix;

GeneralDifferentiableSolver::~GeneralDifferentiableSolver()
{

}
	
double GeneralDifferentiableSolver::solve(Matrix& inputs, const CostAndGradientFunction& callBack)
{
	MatrixVector inputSet({inputs});
	
	return solve(inputSet, callBack);
}

}

}

