/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the LinearSolver class 
 */

#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix; } }

namespace minerva
{

namespace optimizer
{

class LinearSolver
{
public:
	typedef std::vector<matrix::BlockSparseMatrix> BlockSparseMatrixVector;
	
	class CostAndGradient;

public:
	virtual ~LinearSolver();

public:
	/*! \brief Performs unconstrained linear optimization on a differentiable
		function.
	
		\input inputs - The initial parameter values being optimized.
		\input callBack - A CostAndGradient object that is used
			by the optimization library to determine the gradient and
			cost of new parameter values.
	
		\return A floating point value representing the final cost.
	 */
	virtual float solve(BlockSparseMatrixVector& inputs, const CostAndGradient& callBack) = 0;

public:
	class SparseMatrixFormat
	{
	public:
		explicit SparseMatrixFormat(size_t blocks = 0, size_t rowsPerBlock = 0, size_t columnsPerBlock = 0);
		SparseMatrixFormat(const BlockSparseMatrix& );
		
	public:
		size_t blocks;
		size_t rowsPerBlock;
		size_t columnsPerBlock;
	};
	
	typedef std::vector<SparseMatrixFormat> DataStructureFormat;

	class CostAndGradient
	{
	public:
		CostAndGradient(float initialCost = 0.0f, float costReductionFactor = 0.0f,
			const DataStructureFormat& format = DataStructureFormat());
		virtual ~CostAndGradient();
	
	public:
		virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
			const BlockSparseMatrix& inputs) const = 0;

	public:
		BlockSparseMatrixVector getUninitializedDataStructure() const;
	
	public:
		/*! \brief The initial cost at the time the routine is called, can be ignored (set to 0.0f) */
		float initialCost;
		/*! \brief The stopping condition for the solver */
		float costReductionFactor;

	public:
		/*! \brief Structural parameters of the data structure */
		DataStructureFormat format;
	
	};


};

}

}

