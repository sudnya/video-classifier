/*	\file   SumOfSquaresCostFunction.cpp
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SumOfSquaresCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SumOfSquaresCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

namespace minerva
{

namespace network
{

SumOfSquaresCostFunction::~SumOfSquaresCostFunction()
{

}

matrix::Matrix SumOfSquaresCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
	auto difference = apply(output, reference, matrix::Subtract());

	size_t samples = output.size()[0];

	return apply(difference, matrix::SquareAndScale(0.5*samples));
}

matrix::Matrix SumOfSquaresCostFunction::computeDelta(const Matrix& output, const Matrix& reference) const
{
	return apply(output, reference, matrix::Subtract());
}

CostFunction* SumOfSquaresCostFunction::clone() const
{
	return new SumOfSquaresCostFunction;
}

}

}

