/*	\file   SoftmaxCostFunction.cpp
	\date   November 19, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the SoftmaxCostFunction class.
*/

// Minerva Includes
#include <minerva/network/interface/SoftmaxCostFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/Operation.h>

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
	auto normalizedOutput = broadcast(output, reduce(output, {1}, matrix::Maximum()), {1}, matrix::Subtract());

	auto expOutput = apply(normalizedOutput, matrix::Exp());
	
	auto sums = reduce(expOutput, {1}, matrix::Add());
	
	return broadcast(expOutput, sums, {1}, matrix::Divide());
}

Matrix SoftmaxCostFunction::computeCost(const Matrix& output, const Matrix& reference) const
{
	auto softmaxResult = softmax(output);

	auto result = apply(softmaxResult, matrix::Log());

	return apply(apply(reference, result, matrix::Multiply()), matrix::Negate());
}

Matrix SoftmaxCostFunction::computeDelta(const Matrix& output, const Matrix& reference) const
{
	return apply(softmax(output), reference, matrix::Subtract());
}

CostFunction* SoftmaxCostFunction::clone() const
{
	return new SoftmaxCostFunction;
}

}

}



