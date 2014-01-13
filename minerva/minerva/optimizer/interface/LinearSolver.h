/* Author: Sudnya Padalikar
 * Date  : 08/17/2013
 * The interface for the LinearSolver class 
 */

#pragma once

// Standard Library Includes
#include <vector>

// Forward Declarations
namespace minerva { namespace matrix    { class Matrix;             } }
namespace minerva { namespace matrix    { class BlockSparseMatrix;  } }
namespace minerva { namespace optimizer { class SparseMatrixFormat; } }

namespace minerva
{

namespace optimizer
{

class LinearSolver
{
public:
	typedef matrix::BlockSparseMatrix BlockSparseMatrix;
	typedef matrix::Matrix Matrix;
	typedef std::vector<BlockSparseMatrix> BlockSparseMatrixVector;
	
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
	float solve(Matrix& inputs, const CostAndGradient& callBack);
	float solve(BlockSparseMatrix& inputs, const CostAndGradient& callBack);

public:
	typedef std::vector<SparseMatrixFormat> DataStructureFormat;
	
	class CostAndGradient
	{
	public:
		CostAndGradient(float initialCost = 0.0f, float costReductionFactor = 0.0f,
			const DataStructureFormat& format = DataStructureFormat());
		CostAndGradient(float initialCost, float costReductionFactor,
			const BlockSparseMatrixVector& format);
		CostAndGradient(float initialCost, float costReductionFactor,
			const Matrix& format);
		virtual ~CostAndGradient();
	
	public:
		virtual float computeCostAndGradient(BlockSparseMatrixVector& gradient,
			const BlockSparseMatrixVector& inputs) const = 0;

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

class SparseMatrixFormat
{
public:
	explicit SparseMatrixFormat(size_t blocks = 0, size_t rowsPerBlock = 0,
		size_t columnsPerBlock = 0, bool isRowSparse = true);
	SparseMatrixFormat(const matrix::BlockSparseMatrix& );
	SparseMatrixFormat(const matrix::Matrix& );
	
public:
	size_t blocks;
	size_t rowsPerBlock;
	size_t columnsPerBlock;
	bool   isRowSparse;
};

}

}

