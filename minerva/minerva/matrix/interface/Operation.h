/*	\file   Operation.h
	\date   Sunday August 11, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the Operation classes.
*/

#pragma once

// Minerva Includes
#include <minerva/parallel/interface/cuda.h>

// Standard Library Includes
#include <cmath>

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

class Divide : public Operation
{
public:
	CUDA_DECORATOR Divide() : Operation(Operation::Divide), _value(0.0)
	{

	}

	CUDA_DECORATOR Divide(double d) : Operation(Operation::Divide), _value(d)
	{
		
	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l, const T& r) const
	{
		return l / r;
	}
	
	template<typename T>
	CUDA_DECORATOR T operator()(const T& r) const
	{
		return r / _value;
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

class RectifiedLinear : public Operation
{
public:
	CUDA_DECORATOR RectifiedLinear() : Operation(Operation::RectifiedLinear)
	{

	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l) const
	{
		return l > 20.0 ? 20.0 : l < 0.0 ? 0.0 : l;
	}
	
};

class RectifiedLinearDerivative : public Operation
{
public:
	CUDA_DECORATOR RectifiedLinearDerivative() : Operation(Operation::RectifiedLinearDerivative)
	{

	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l) const
	{
		return l > 20.0 ? 0.0 : l < 0.0 ? 0.0 : 1.0;
	}
	
};

class Sigmoid : public Operation
{
public:
	CUDA_DECORATOR Sigmoid() : Operation(Operation::Sigmoid)
	{

	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l) const
	{
		return T(1.0) / (T(1.0) + T(std::exp(-l)));
	}
	
};

class SigmoidDerivative : public Operation
{
public:
	CUDA_DECORATOR SigmoidDerivative() : Operation(Operation::SigmoidDerivative)
	{

	}

public:
	template<typename T>
	CUDA_DECORATOR T operator()(const T& l) const
	{
		return 1.0 - matrix::Sigmoid()(l);
	}
	
};

typedef std::tuple<Add, Multiply, Fill, Divide, RectifiedLinear,
				   RectifiedLinearDerivative, Sigmoid, SigmoidDerivative> AllOperations;

typedef std::tuple<Add, Multiply, Divide> AllBinaryOperations;

typedef std::tuple<Add, Multiply, Fill, Divide, RectifiedLinear, RectifiedLinearDerivative,
				   Sigmoid, SigmoidDerivative> AllUnaryOperations;

}
}

