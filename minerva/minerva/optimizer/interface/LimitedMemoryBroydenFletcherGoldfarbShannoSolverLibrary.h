/*	\file   LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary.h
	\date   Thursday August 15, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the
		LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary class.
*/

#pragma once


// Forward Declarations
namespace minerva { namespace video { class CvCapture; } }

namespace minerva
{

namespace optimizer
{

class LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary
{
public:
	static void load();
	static bool loaded();

public:
	typedef double (*lbfgs_evaluate_t)(void* instance, const double* x,
		double* g, const int n, const double step);
	typedef int (*lbfgs_progress_t )(void* instance, const double* x,
		const double* g, const double fx, const double xnorm,
		const double gnorm, const double step, int n, int k, int ls);

	class lbfgs_parameter_t;

public:
	/*
	Start a L-BFGS optimization.

	Parameters:
    	n 	The number of variables.
    	x 	The array of variables. A client program can set default values for
    		the optimization and receive the optimization result through this
    		array. This array must be allocated by lbfgs_malloc function for
    		libLBFGS built with SSE/SSE2 optimization routine enabled. The
    		library built without SSE/SSE2 optimization does not have such a
    		requirement.
    	
    	ptr_fx 	The pointer to the variable that receives the final value of
    		the objective function for the variables. This argument can be set
    		to NULL if the final value of the objective function is unnecessary.
    	proc_evaluate 	The callback function to provide function and gradient
    		evaluations given a current values of variables. A client program
    		must implement a callback function compatible with lbfgs_evaluate_t
    		and pass the pointer to the callback function.
    	proc_progress 	The callback function to receive the progress
    		(the number of iterations, the current value of the objective
    		function) of the minimization process. This argument can be set
    		to NULL if a progress report is unnecessary.
    	instance 	A user data for the client program. The callback functions
    		will receive the value of this argument.
    	param 	The pointer to a structure representing parameters for L-BFGS
    		optimization. A client program can set this parameter to NULL to
    		use the default parameters. Call lbfgs_parameter_init() function
    		to fill a structure with the default values.

		Return values:
    		int  The status code. This function returns zero if the minimization
    			process terminates without an error. A non-zero value
    			indicates an error. 
	*/
	static int lbfgs(int n, double* x, double* ptr_fx,
		lbfgs_evaluate_t proc_evaluate, lbfgs_progress_t proc_progress,
		void* instance, lbfgs_parameter_t* param);
	
	static void   lbfgs_parameter_init(lbfgs_parameter_t* param);
	static double* lbfgs_malloc(int n);
	static void   lbfgs_free(double* x);

public:
	class lbfgs_parameter_t
	{
	public:
		/**
		* The number of corrections to approximate the inverse hessian matrix.
		* The L-BFGS routine stores the computation results of previous \ref m
		* iterations to approximate the inverse hessian matrix of the current
		* iteration. This parameter controls the size of the limited memories
		* (corrections). The default value is \c 6. Values less than \c 3 are
		* not recommended. Large values will result in excessive computing time.
		*/
			int m;

			/**
		* Epsilon for convergence test.
		* This parameter determines the accuracy with which the solution is to
		* be found. A minimization terminates when
		* ||g|| < \ref epsilon * max(1, ||x||),
		* where ||.|| denotes the Euclidean (L2) norm. The default value is
		* \c 1e-5.
		*/
			double epsilon;

			/**
		* Distance for delta-based convergence test.
		* This parameter determines the distance, in iterations, to compute
		* the rate of decrease of the objective function. If the value of this
		* parameter is zero, the library does not perform the delta-based
		* convergence test. The default value is \c 0.
		*/
			int past;

			/**
		* Delta for convergence test.
		* This parameter determines the minimum rate of decrease of the
		* objective function. The library stops iterations when the
		* following condition is met:
		* (f' - f) / f < \ref delta,
		* where f' is the objective value of \ref past iterations ago, and f is
		* the objective value of the current iteration.
		* The default value is \c 0.
		*/
			double delta;

			/**
		* The maximum number of iterations.
		* The lbfgs() function terminates an optimization process with
		* ::LBFGSERR_MAXIMUMITERATION status code when the iteration count
		* exceedes this parameter. Setting this parameter to zero continues an
		* optimization process until a convergence or error. The default value
		* is \c 0.
		*/
			int max_iterations;

			/**
		* The line search algorithm.
		* This parameter specifies a line search algorithm to be used by the
		* L-BFGS routine.
		*/
			int linesearch;

			/**
		* The maximum number of trials for the line search.
		* This parameter controls the number of function and gradients evaluations
		* per iteration for the line search routine. The default value is \c 20.
		*/
			int max_linesearch;

			/**
		* The minimum step of the line search routine.
		* The default value is \c 1e-20. This value need not be modified unless
		* the exponents are too large for the machine being used, or unless the
		* problem is extremely badly scaled (in which case the exponents should
		* be increased).
		*/
			double min_step;

			/**
		* The maximum step of the line search.
		* The default value is \c 1e+20. This value need not be modified unless
		* the exponents are too large for the machine being used, or unless the
		* problem is extremely badly scaled (in which case the exponents should
		* be increased).
		*/
			double max_step;

			/**
		* A parameter to control the accuracy of the line search routine.
		* The default value is \c 1e-4. This parameter should be greater
		* than zero and smaller than \c 0.5.
		*/
			double ftol;

			/**
		* A coefficient for the Wolfe condition.
		* This parameter is valid only when the backtracking line-search
		* algorithm is used with the Wolfe condition,
		* ::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE or
		* ::LBFGS_LINESEARCH_BACKTRACKING_WOLFE .
		* The default value is \c 0.9. This parameter should be greater
		* the \ref ftol parameter and smaller than \c 1.0.
		*/
			double wolfe;

			/**
		* A parameter to control the accuracy of the line search routine.
		* The default value is \c 0.9. If the function and gradient
		* evaluations are inexpensive with respect to the cost of the
		* iteration (which is sometimes the case when solving very large
		* problems) it may be advantageous to set this parameter to a small
		* value. A typical small value is \c 0.1. This parameter shuold be
		* greater than the \ref ftol parameter (\c 1e-4) and smaller than
		* \c 1.0.
		*/
			double gtol;

			/**
		* The machine precision for doubleing-point values.
		* This parameter must be a positive value set by a client program to
		* estimate the machine precision. The line search routine will terminate
		* with the status code (::LBFGSERR_ROUNDING_ERROR) if the relative width
		* of the interval of uncertainty is less than this parameter.
		*/
			double xtol;

			/**
		* Coeefficient for the L1 norm of variables.
		* This parameter should be set to zero for standard minimization
		* problems. Setting this parameter to a positive value activates
		* Orthant-Wise Limited-memory Quasi-Newton (OWL-QN) method, which
		* minimizes the objective function F(x) combined with the L1 norm |x|
		* of the variables, {F(x) + C |x|}. This parameter is the coeefficient
		* for the |x|, i.e., C. As the L1 norm |x| is not differentiable at
		* zero, the library modifies function and gradient evaluations from
		* a client program suitably; a client program thus have only to return
		* the function value F(x) and gradients G(x) as usual. The default value
		* is zero.
		*/
			double orthantwise_c;

			/**
		* Start index for computing L1 norm of the variables.
		* This parameter is valid only for OWL-QN method
		* (i.e., \ref orthantwise_c != 0). This parameter b (0 <= b < N)
		* specifies the index number from which the library computes the
		* L1 norm of the variables x,
		* |x| := |x_{b}| + |x_{b+1}| + ... + |x_{N}| .
		* In other words, variables x_1, ..., x_{b-1} are not used for
		* computing the L1 norm. Setting b (0 < b < N), one can protect
		* variables, x_1, ..., x_{b-1} (e.g., a bias term of logistic
		* regression) from being regularized. The default value is zero.
		*/
			int orthantwise_start;

			/**
		* End index for computing L1 norm of the variables.
		* This parameter is valid only for OWL-QN method
		* (i.e., \ref orthantwise_c != 0). This parameter e (0 < e <= N)
		* specifies the index number at which the library stops computing the
		* L1 norm of the variables x,
		*/
		int orthantwise_end;
	};

	enum {
		/** L-BFGS reaches convergence. */
		LBFGS_SUCCESS = 0,
		LBFGS_CONVERGENCE = 0,
		LBFGS_STOP,
		/** The initial variables already minimize the objective function. */
		LBFGS_ALREADY_MINIMIZED,

		/** Unknown error. */
		LBFGSERR_UNKNOWNERROR = -1024,
		/** Logic error. */
		LBFGSERR_LOGICERROR = -1023,
		/** Insufficient memory. */
		LBFGSERR_OUTOFMEMORY = -1022,
		/** The minimization process has been canceled. */
		LBFGSERR_CANCELED = -1021,
		/** Invalid number of variables specified. */
		LBFGSERR_INVALID_N = -1020,
		/** Invalid number of variables (for SSE) specified. */
		LBFGSERR_INVALID_N_SSE = -1019,
		/** The array x must be aligned to 16 (for SSE). */
		LBFGSERR_INVALID_X_SSE = -1018,
		/** Invalid parameter lbfgs_parameter_t::epsilon specified. */
		LBFGSERR_INVALID_EPSILON = -1017,
		/** Invalid parameter lbfgs_parameter_t::past specified. */
		LBFGSERR_INVALID_TESTPERIOD = -1016,
		/** Invalid parameter lbfgs_parameter_t::delta specified. */
		LBFGSERR_INVALID_DELTA = -1015,
		/** Invalid parameter lbfgs_parameter_t::linesearch specified. */
		LBFGSERR_INVALID_LINESEARCH = -1014,
		/** Invalid parameter lbfgs_parameter_t::max_step specified. */
		LBFGSERR_INVALID_MINSTEP = -1013,
		/** Invalid parameter lbfgs_parameter_t::max_step specified. */
		LBFGSERR_INVALID_MAXSTEP = -1012,
		/** Invalid parameter lbfgs_parameter_t::ftol specified. */
		LBFGSERR_INVALID_FTOL = -1011,
		/** Invalid parameter lbfgs_parameter_t::wolfe specified. */
		LBFGSERR_INVALID_WOLFE = -1010,
		/** Invalid parameter lbfgs_parameter_t::gtol specified. */
		LBFGSERR_INVALID_GTOL = -1009,
		/** Invalid parameter lbfgs_parameter_t::xtol specified. */
		LBFGSERR_INVALID_XTOL = -1008,
		/** Invalid parameter lbfgs_parameter_t::max_linesearch specified. */
		LBFGSERR_INVALID_MAXLINESEARCH = -1007,
		/** Invalid parameter lbfgs_parameter_t::orthantwise_c specified. */
		LBFGSERR_INVALID_ORTHANTWISE = -1006,
		/** Invalid parameter lbfgs_parameter_t::orthantwise_start specified. */
		LBFGSERR_INVALID_ORTHANTWISE_START = -1005,
		/** Invalid parameter lbfgs_parameter_t::orthantwise_end specified. */
		LBFGSERR_INVALID_ORTHANTWISE_END = -1004,
		/** The line-search step went out of the interval of uncertainty. */
		LBFGSERR_OUTOFINTERVAL = -1003,
		/** A logic error occurred; alternatively, the interval of uncertainty became too small. */
		LBFGSERR_INCORRECT_TMINMAX = -1002,
		/** A rounding error occurred; alternatively, no line-search step satisfies the sufficient decrease and curvature conditions. */
		LBFGSERR_ROUNDING_ERROR = -1001,
		/** The line-search step became smaller than lbfgs_parameter_t::min_step. */
		LBFGSERR_MINIMUMSTEP = -1000,
		/** The line-search step became larger than lbfgs_parameter_t::max_step. */
		LBFGSERR_MAXIMUMSTEP = -999,
		/** The line-search routine reaches the maximum number of evaluations. */
		LBFGSERR_MAXIMUMLINESEARCH = -998,
		/** The algorithm routine reaches the maximum number of iterations. */
		LBFGSERR_MAXIMUMITERATION = -997,
		/** Relative width of the interval of uncertainty is at most lbfgs_parameter_t::xtol. */
		LBFGSERR_WIDTHTOOSMALL = -996,
		/** A logic error (negative line-search step) occurred. */
		LBFGSERR_INVALIDPARAMETERS = -995,
		/** The current search direction increases the objective function value. */
		LBFGSERR_INCREASEGRADIENT = -994,
	};

private:
	static void _check();
	
private:
	class Interface
	{
	public:
		int (*lbfgs)(int n, double* x, double* ptr_fx,
			lbfgs_evaluate_t proc_evaluate, lbfgs_progress_t proc_progress,
			void* instance, lbfgs_parameter_t* param);
		void (*lbfgs_parameter_init)(lbfgs_parameter_t* param);
		double* (*lbfgs_malloc)(int n);
		void (*lbfgs_free)(double* x);
	
    public:
		/*! \brief The constructor zeros out all of the pointers*/
		Interface();
		
		/*! \brief The destructor closes dlls */
		~Interface();
		/*! \brief Load the library */
		void load();
		/*! \brief Has the library been loaded? */
		bool loaded() const;
		/*! \brief unloads the library */
		void unload();
				
	private:
		
		void* _library;
	};
	
private:
	static Interface _interface;

};

typedef LimitedMemoryBroydenFletcherGoldfarbShannoSolverLibrary
	LBFGSSolverLibrary;

}

}


