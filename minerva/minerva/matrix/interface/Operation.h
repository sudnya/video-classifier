/*	\file   Operation.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Operation classes.
*/

#pragma once

#include <minerva/parallel/interface/cuda.h>

namespace minerva
{
namespace matrix
{


/*! \brief A class for specifying basic matrix operations. */
class Operation
{
public:
	enum Type
	{
		Add,
		Subtract,
		Multiply,
		Divide,
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
		NotEqual,
		Fill		
	};

public:
	Operation(Type t);

public:
	bool operator==(const Operation&) const;

};

class Add : public Operation
{
public:
	CUDA_DECORATOR Add() : Operation(Operation::Add), _value(0.0)
	{

	}

	CUDA_DECORATOR Add(double d) : Operation(Operation::Add), _value(d)
	{
		
	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l, const T& r) const
	{
		return l + r;
	}
	
	template<typename T>
	CUDA_DECORATOR T operator()(const T& r) const
	{
		return _value + r;
	}
	
private:
	double _value;

};

class Multiply : public Operation
{
public:
	CUDA_DECORATOR Multiply() : Operation(Operation::Multiply), _value(0.0)
	{

	}

	CUDA_DECORATOR Multiply(double d) : Operation(Operation::Multiply), _value(d)
	{
		
	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l, const T& r) const
	{
		return l * r;
	}
	
	template<typename T>
	CUDA_DECORATOR T operator()(const T& r) const
	{
		return _value * r;
	}
	
private:
	double _value;

};

class Fill : public Operation
{
public:
	CUDA_DECORATOR Fill(double d) : Operation(Operation::Fill), _value(d)
	{
		
	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& r) const
	{
		return _value;
	}
	
private:
	double _value;

};

typedef std::tuple<Add, Multiply> AllOperations;

typedef std::tuple<Add, Multiply> AllBinaryOperations;

typedef std::tuple<Add, Multiply, Fill> AllUnaryOperations;

}
}

