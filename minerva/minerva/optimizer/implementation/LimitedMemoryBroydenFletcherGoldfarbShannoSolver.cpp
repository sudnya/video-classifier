/*	\file   LimitedMemoryBroydenFletcherGoldfarbShannoSolver.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LimitedMemoryBroydenFletcherGoldfarbShannoSolver class.
*/

// Minvera Includes
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolver.h>
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary.h>

#include <minerva/matrix/interface/Matrix.h>

// Standard Library Includes
#include <stdexcept>

namespace minerva
{

namespace optimizer
{

typedef matrix::Matrix Matrix;
typedef Matrix::FloatVector FloatVector;

typedef LBFGSSolver::CostAndGradient CostAndGradient;

LBFGSSolver::~LimitedMemoryBroydenFletcherGoldfarbShannoSolver()
{

}

static void copyData(float* data, const Matrix& matrix)
{
	auto matrixData = matrix.data();
	
	std::memcpy(data, matrixData.data(), matrix.size() * sizeof(float));
}

static float lbfgsCallback(void* instance, const float* x, float* g,
	const int n, const float step)
{
	const CostAndGradient* callback = reinterpret_cast<const CostAndGradient*>(instance);

	// TODO: get rid of these copies	
	Matrix weights (1, n, FloatVector(x, x + n));
	Matrix gradient(1, n);
	
	float cost = callback->computeCostAndGradient(gradient, weights);
	
	auto data = gradient.data();
	
	std::memcpy(g, data.data(), sizeof(float) * data.size());
	
	return cost;
}

float LBFGSSolver::solve(Matrix& inputs, const CostAndGradient& callback)
{
	float* inputArray = LBFGSSolverLibrary::lbfgs_malloc(inputs.size());
	
	if(inputArray == nullptr)
	{
		throw std::runtime_error("Failed to allocate memory for LBFGSSolver");		
	}
	
	copyData(inputArray, inputs);
	
	float finalCost = 0.0f;
	
	int status = LBFGSSolverLibrary::lbfgs(inputs.size(), inputArray,
		&finalCost, lbfgsCallback, nullptr,
		const_cast<CostAndGradient*>(&callback), nullptr);
	
	LBFGSSolverLibrary::lbfgs_free(inputArray);
	
	if(status != 0)
	{
		throw std::runtime_error("LBFGSSolver returned error code.");		
	}

	return finalCost;
}

bool LBFGSSolver::isSupported()
{
	LBFGSSolverLibrary::load();
	
	return LBFGSSolverLibrary::loaded();
}

}

}

