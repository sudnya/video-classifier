/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The implementation for the GeneralDifferentiableSolver class 
 */

// Lucius Includes
#include <lucius/optimizer/interface/GeneralDifferentiableSolver.h>

#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/Matrix.h>

namespace lucius
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

