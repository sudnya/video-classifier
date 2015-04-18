/*    \file   Operation.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the Operation classes.
*/

#pragma once

// Minerva Includes
#include <minerva/parallel/interface/cuda.h>

// Standard Library Includes
#include <cmath>
#include <algorithm>

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
        Maximum,
        Minimum,
        Equal,
        LessThan,
        NotEqual,
        LessThanOrEqual,
        GreaterThanOrEqual,
        Fill,
        Square,
        SquareAndScale,
        Copy
    };

public:
    Operation(Type t);

public:
    bool operator==(const Operation&) const;

private:
    Type _type;

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

class Subtract : public Operation
{
public:
    CUDA_DECORATOR Subtract() : Operation(Operation::Subtract), _value(0.0)
    {

    }

    CUDA_DECORATOR Subtract(double d) : Operation(Operation::Subtract), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return l - r;
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r - _value;
    }

private:
    double _value;

};

class Log : public Operation
{
public:
    CUDA_DECORATOR Log() : Operation(Operation::Log)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return std::log(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return std::logf(l);
    }
};

class Exp : public Operation
{
public:
    CUDA_DECORATOR Exp() : Operation(Operation::Exp)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return std::exp(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return std::expf(l);
    }
};

class Abs : public Operation
{
public:
    CUDA_DECORATOR Abs() : Operation(Operation::Abs)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return std::abs(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return std::fabs(l);
    }
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
        return (l > 20.0) ? 20.0 : ((l < 0.0) ? 0.0 : l);
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
        return (l >= 20.0) ? 0.0 : ((l <= 0.0) ? 0.0 : 1.0);
    }

};

class Negate : public Operation
{
public:
    CUDA_DECORATOR Negate() : Operation(Operation::Negate)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return -l;
    }

};

class Maximum : public Operation
{
public:
    CUDA_DECORATOR Maximum() : Operation(Operation::Maximum), _value(0.0)
    {

    }

    CUDA_DECORATOR Maximum(double d) : Operation(Operation::Maximum), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return std::max(l, r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return std::max(T(_value), r);
    }

private:
    double _value;

};

class Minimum : public Operation
{
public:
    CUDA_DECORATOR Minimum() : Operation(Operation::Minimum), _value(0.0)
    {

    }

    CUDA_DECORATOR Minimum(double d) : Operation(Operation::Minimum), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return std::min(l, r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return std::min(T(_value), r);
    }

private:
    double _value;

};

class Equal : public Operation
{
public:
    CUDA_DECORATOR Equal() : Operation(Operation::Equal), _value(0.0)
    {

    }

    CUDA_DECORATOR Equal(double d) : Operation(Operation::Equal), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l == r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(_value == r);
    }

private:
    double _value;

};

class LessThan : public Operation
{
public:
    CUDA_DECORATOR LessThan() : Operation(Operation::LessThan), _value(0.0)
    {

    }

    CUDA_DECORATOR LessThan(double d) : Operation(Operation::LessThan), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l < r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(r < _value);
    }

private:
    double _value;

};

class NotEqual : public Operation
{
public:
    CUDA_DECORATOR NotEqual() : Operation(Operation::NotEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR NotEqual(double d) : Operation(Operation::NotEqual), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l != r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(_value != r);
    }

private:
    double _value;

};

class GreaterThanOrEqual : public Operation
{
public:
    CUDA_DECORATOR GreaterThanOrEqual() : Operation(Operation::GreaterThanOrEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR GreaterThanOrEqual(double d) : Operation(Operation::GreaterThanOrEqual), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l >= r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(r >= _value);
    }

private:
    double _value;

};

class LessThanOrEqual : public Operation
{
public:
    CUDA_DECORATOR LessThanOrEqual() : Operation(Operation::LessThanOrEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR LessThanOrEqual(double d) : Operation(Operation::LessThanOrEqual), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l <= r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(r <= _value);
    }

private:
    double _value;

};

class Fill : public Operation
{
public:
    CUDA_DECORATOR Fill(double d = 0.0) : Operation(Operation::Fill), _value(d)
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

class Square : public Operation
{
public:
    CUDA_DECORATOR Square() : Operation(Operation::Square)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r * r;
    }

};

class SquareAndScale : public Operation
{
public:
    CUDA_DECORATOR SquareAndScale(double d = 1.0) : Operation(Operation::SquareAndScale), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r * r * _value;
    }

private:
    double _value;

};

typedef std::tuple<Add, Subtract, Multiply, Divide, Log, Exp, Abs, RectifiedLinear,
                   RectifiedLinearDerivative, Sigmoid, SigmoidDerivative, Negate, Maximum,
                   Minimum, Equal, LessThan, NotEqual, Fill, Square, SquareAndScale> AllOperations;

typedef std::tuple<Add, Subtract, Multiply, Divide, Maximum, Minimum,
                   Equal, LessThan, NotEqual> AllBinaryOperations;

typedef std::tuple<Add, Subtract, Multiply, Divide, Log, Exp, Abs, RectifiedLinear,
                   RectifiedLinearDerivative, Sigmoid, SigmoidDerivative, Negate, Maximum,
                   Minimum, Equal, LessThan, NotEqual, Fill, Square, SquareAndScale> AllUnaryOperations;

}
}

