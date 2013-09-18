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
	typedef float (*lbfgs_evaluate_t)(void* instance, const float* x, float* g,
		const int n, const float step);
	typedef int (*lbfgs_progress_t )(void* instance, const float* x,
		const float* g, const float fx, const float xnorm, const float gnorm,
		const float step, int n, int k, int ls);

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
	static int lbfgs(int n, float* x, float* ptr_fx,
		lbfgs_evaluate_t proc_evaluate, lbfgs_progress_t proc_progress,
		void* instance, lbfgs_parameter_t* param);
	
	static void   lbfgs_parameter_init(lbfgs_parameter_t* param);
	static float* lbfgs_malloc(int n);
	static void   lbfgs_free(float* x);

private:
	static void _check();
	
private:
	class Interface
	{
	public:
		int (*lbfgs)(int n, float* x, float* ptr_fx,
			lbfgs_evaluate_t proc_evaluate, lbfgs_progress_t proc_progress,
			void* instance, lbfgs_parameter_t* param);
		void (*lbfgs_parameter_init)(lbfgs_parameter_t* param);
		float* (*lbfgs_malloc)(int n);
		void (*lbfgs_free)(float* x);
	
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


