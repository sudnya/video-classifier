/*	\file   LimitedMemoryBroydenFletcherGoldfarbShannoSolver.cpp
	\date   Saturday August 10, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LimitedMemoryBroydenFletcherGoldfarbShannoSolver class.
*/

// Minvera Includes
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolver.h>
#include <minerva/optimizer/interface/LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>


// Standard Library Includes
#include <stdexcept>
#include <sstream>
#include <limits>

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

static void copyData(double* data, const Matrix& matrix)
{
	std::vector<double> matrixDataAsDoubles(matrix.begin(),
		matrix.end());
	
	std::memcpy(data, matrixDataAsDoubles.data(),
		matrix.size() * sizeof(double));
}

static void copyData(Matrix& matrix, const double* data)
{
	matrix.data() = FloatVector(data, matrix.size() + data);
}

static double lbfgsCallback(void* instance, const double* x, double* g,
	const int n, const double step)
{
	const CostAndGradient* callback =
		reinterpret_cast<const CostAndGradient*>(instance);

	// TODO: get rid of these copies	
	Matrix weights (1, n, FloatVector(x, x + n));
	Matrix gradient(1, n);
	
	float cost = callback->computeCostAndGradient(gradient, weights);
	
	copyData(g, gradient);
	
	return cost;
}

static int lbfgsProgress(void* instance, const double* x,
	const double* g, const double fx, const double xnorm, const double gnorm,
	const double step, int n, int k, int ls)
{
	const CostAndGradient* callback =
		reinterpret_cast<const CostAndGradient*>(instance);

	util::log("LBFGSSolver") << "LBFGS Update (cost " << fx << ", xnorm "
		<< xnorm << ", gnorm " << gnorm << ", step " << step << ", n " << n
		<< ", k " << k << ", ls " << ls << ")\n";

	// Early exit if cost is low enough
	if(fx < callback->initialCost * callback->costReductionFactor)
	{
		return 1;
	}

	return 0;
}

static std::string getMessage(int status)
{
	switch(status)
	{
	case LBFGSSolverLibrary::LBFGS_SUCCESS:
		return "L-BFGS reaches convergence";
	case LBFGSSolverLibrary::LBFGS_STOP:
		return "L-BFGS stopped by user";
	case LBFGSSolverLibrary::LBFGS_ALREADY_MINIMIZED:
		return "The initial variables already minimize the objective function.";
	case LBFGSSolverLibrary::LBFGSERR_UNKNOWNERROR:
		return "Unknown error.";
	case LBFGSSolverLibrary::LBFGSERR_LOGICERROR:
		return "Logic error.";
	case LBFGSSolverLibrary::LBFGSERR_OUTOFMEMORY:
		return "Insufficient memory.";
	case LBFGSSolverLibrary::LBFGSERR_CANCELED:
		return "The minimization process has been canceled.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_N:
		return "Invalid number of variables specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_N_SSE:
		return "Invalid number of variables (for SSE) specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_X_SSE:
		return "The array x must be aligned to 16 (for SSE).";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_EPSILON:
		return "Invalid parameter lbfgs_parameter_t::epsilon specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_TESTPERIOD:
		return "Invalid parameter lbfgs_parameter_t::past specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_DELTA:
		return "Invalid parameter lbfgs_parameter_t::delta specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_LINESEARCH:
		return "Invalid parameter lbfgs_parameter_t::linesearch specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_MINSTEP:
		return "Invalid parameter lbfgs_parameter_t::max_step specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_MAXSTEP:
		return "Invalid parameter lbfgs_parameter_t::max_step specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_FTOL:
		return "Invalid parameter lbfgs_parameter_t::ftol specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_WOLFE:
		return "Invalid parameter lbfgs_parameter_t::wolfe specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_GTOL:
		return "Invalid parameter lbfgs_parameter_t::gtol specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_XTOL:
		return "Invalid parameter lbfgs_parameter_t::xtol specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_MAXLINESEARCH:
		return "Invalid parameter lbfgs_parameter_t::max_linesearch specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_ORTHANTWISE:
		return "Invalid parameter lbfgs_parameter_t::orthantwise_c specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_ORTHANTWISE_START:
		return "Invalid parameter lbfgs_parameter_t::orthantwise_start "
			"specified.";
	case LBFGSSolverLibrary::LBFGSERR_INVALID_ORTHANTWISE_END:
		return "Invalid parameter lbfgs_parameter_t::orthantwise_end "
			"specified.";
	case LBFGSSolverLibrary::LBFGSERR_OUTOFINTERVAL:
		return "The line-search step went out of the interval of uncertainty.";
	case LBFGSSolverLibrary::LBFGSERR_INCORRECT_TMINMAX:
		return "A logic error occurred; alternatively, the interval of"
			" uncertainty became too small.";
	case LBFGSSolverLibrary::LBFGSERR_ROUNDING_ERROR:
		return "A rounding error occurred; alternatively, no line-search "
			"step satisfies the sufficient decrease and curvature conditions.";
	case LBFGSSolverLibrary::LBFGSERR_MINIMUMSTEP:
		return "The line-search step became smaller than "
			"lbfgs_parameter_t::min_step.";
	case LBFGSSolverLibrary::LBFGSERR_MAXIMUMSTEP:
		return "The line-search step became larger than "
			"lbfgs_parameter_t::max_step.";
	case LBFGSSolverLibrary::LBFGSERR_MAXIMUMLINESEARCH:
		return "The line-search routine reaches the maximum "
			"number of evaluations.";
	case LBFGSSolverLibrary::LBFGSERR_MAXIMUMITERATION:
		return "The algorithm routine reaches the maximum number "
			"of iterations.";
	case LBFGSSolverLibrary::LBFGSERR_WIDTHTOOSMALL:
		return "Relative width of the interval of uncertainty is at "
			"most lbfgs_parameter_t::xtol.";
	case LBFGSSolverLibrary::LBFGSERR_INVALIDPARAMETERS:
		return "A logic error (negative line-search step) occurred.";
	case LBFGSSolverLibrary::LBFGSERR_INCREASEGRADIENT:
		return "The current search direction increases the objective "
			"function value.";
	}
	
	return "unknown error";
}

float LBFGSSolver::solve(Matrix& inputs, const CostAndGradient& callback)
{
	double* inputArray = LBFGSSolverLibrary::lbfgs_malloc(inputs.size());
	
	if(inputArray == nullptr)
	{
		throw std::runtime_error("Failed to allocate memory for LBFGSSolver");		
	}
	
	copyData(inputArray, inputs);
	
	double finalCost = 0.0;
	
	LBFGSSolverLibrary::lbfgs_parameter_t parameters;
	
	LBFGSSolverLibrary::lbfgs_parameter_init(&parameters);
	
	//parameters.min_step = 10e-10;
	//parameters.max_linesearch = 5.0;
	parameters.linesearch = LBFGSSolverLibrary::LBFGS_LINESEARCH_DEFAULT;//LBFGSSolverLibrary::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE;
	parameters.epsilon = 1e-10;
	//parameters.m = 3;
	//parameters.xtol = 0.000001;
	parameters.ftol = 1e-10;
	//parameters.gtol = 1e-10;
	
	parameters.max_linesearch = util::KnobDatabase::getKnobValue(
		"LBFGSSolver::MaxLineSearchIterations", 50);

	parameters.max_iterations =
		util::KnobDatabase::getKnobValue("LBFGSSolver::MaxIterations", 0);
	
	int status = LBFGSSolverLibrary::lbfgs(inputs.size(), inputArray,
		&finalCost, lbfgsCallback, lbfgsProgress,
		const_cast<CostAndGradient*>(&callback), &parameters);
	
	if(status < 0)
	{
		if(status != LBFGSSolverLibrary::LBFGSERR_MAXIMUMITERATION &&
			status != LBFGSSolverLibrary::LBFGSERR_ROUNDING_ERROR &&
			status != LBFGSSolverLibrary::LBFGSERR_MAXIMUMLINESEARCH)
		{
			LBFGSSolverLibrary::lbfgs_free(inputArray);
		
			std::stringstream code;
		
			code << status;
			
			throw std::runtime_error("LBFGSSolver returned error code (" +
				code.str() + ") message (" + getMessage(status) + ").");		
		}

		util::log("LBFGSSolver") << "Terminated search early "
			<< getMessage(status) << ".\n";
	}
	
	copyData(inputs, inputArray);

	LBFGSSolverLibrary::lbfgs_free(inputArray);
		
	return finalCost;
}

bool LBFGSSolver::isSupported()
{
	LBFGSSolverLibrary::load();
	
	return LBFGSSolverLibrary::loaded();
}

}

}

