/*	\file   CostFunction.h
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the CostFunction class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace network
{

class CostFunction
{
public:
	typedef matrix::Matrix Matrix;

public:
	virtual ~CostFunction();

public:
	/*! \brief Run the cost function on the specified output and reference. */
	virtual Matrix computeCost(const Matrix& output, const Matrix& reference) const = 0;

	/*! \brief Determine the change in the output needed to minimize the cost function. */
	virtual Matrix computeDelta(const Matrix& output, const Matrix& reference) const = 0;

public:
	/*! \brief Clone */
	virtual CostFunction* clone() const = 0;

};

}

}

