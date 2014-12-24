/*	\file   CostFunction.h
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the CostFunction class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrix; } }

namespace minerva
{

namespace neuralnetwork
{

class CostFunction
{
public:
	virtual ~CostFunction();

public:
	/*! \brief Run the cost function on the specified output and reference. */
	virtual BlockSparseMatrix computeCost(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const = 0;

	/*! \brief Determine the change in the output needed to minimize the cost function. */
	virtual BlockSparseMatrix computeDelta(const BlockSparseMatrix& output, const BlockSparseMatrix& reference) const = 0;

};

}

}

