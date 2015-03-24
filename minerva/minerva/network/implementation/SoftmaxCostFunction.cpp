/*	\file   SoftmaxCostFunction.cpp
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SoftmaxCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SoftmaxCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/debug.h>

namespace minerva
{

namespace network
{

typedef matrix::Matrix Matrix;

SoftmaxCostFunction::~SoftmaxCostFunction()
{

}

static Matrix softmax(const Matrix& output)
{
	auto normalizedOutput = broadcast(output, reduce(output, Maximum(), {1}), Subtract());

	auto expOutput = apply(normalizedOutput, Exp());
	
	auto sums = reduce(expOutput, Sum(), {1});
	
	return broadcast(expOutput, sums, Divide());
}

Matrix SoftmaxCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
	auto softmaxResult = softmax(output);

	auto result = apply(Log(), softmaxResult);

	return apply(apply(reference, result, Multiply()), Negate());
}

Matrix SoftmaxCostFunction::computeDelta(const Matrix& output, const Matrix& reference) const
{
	return apply(softmax(output), reference, Subtract());
}

CostFunction* SoftmaxCostFunction::clone() const
{
	return new SoftmaxCostFunction;
}

}

}



