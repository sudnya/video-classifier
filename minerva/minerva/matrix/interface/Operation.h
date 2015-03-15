/*	\file   Operation.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Operation classes.
*/

#pragma once

namespace minerva
{
namespace matrix
{


/*! \brief A class for specifying basic matrix operations. */
class Operation
{
public:
	Operation(enum Type);

public:
	enum Type
	{
		Add,
		Subtract,
		Multiply,
		Log,
		Exp,
		Abs,
		Sigmoid,
		SigmoidDerivative,
		RectifiedLinear,
		RectifiedLinearDerivative,
		KLDivergence,
		KLDivergenceDerivative,
		Negate,
		Max,
		Min,
		Equal,
		LessThan,
		NotEqual
		
	};



};

template<typename T>
class Add : public Operation
{
public:
	Add() : Operation(Operation::Add)
	{

	}

public:
	T operator()(const T& l, const T& r)
	{
		return l + r;
	}
};

typedef std::tuple<Add> AllOperations;

}
}

