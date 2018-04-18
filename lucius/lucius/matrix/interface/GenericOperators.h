/*  \file   GenericOperators.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the GenericOperator classes.
*/

#pragma once

// Lucius Includes
#include <lucius/matrix/interface/StaticOperator.h>

#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/ScalarOperations.h>

#include <lucius/matrix/interface/DimensionTransformations.h>

// Standard Library Includes
#include <cmath>
#include <algorithm>
#include <memory>

namespace lucius
{
namespace matrix
{

class Add : public StaticOperator
{
public:
    CUDA_DECORATOR Add() : StaticOperator(StaticOperator::Add), _value(0.0)
    {

    }

    CUDA_DECORATOR Add(double d) : StaticOperator(StaticOperator::Add), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Add>(*this);
    }

private:
    double _value;

};

class Subtract : public StaticOperator
{
public:
    CUDA_DECORATOR Subtract() : StaticOperator(StaticOperator::Subtract), _value(0.0)
    {

    }

    CUDA_DECORATOR Subtract(double d) : StaticOperator(StaticOperator::Subtract), _value(d)
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
        return _value - r;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Subtract>(*this);
    }

private:
    double _value;

};

class Log : public StaticOperator
{
public:
    CUDA_DECORATOR Log() : StaticOperator(StaticOperator::Log)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return log(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return logf(l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Log>(*this);
    }
};

class Exp : public StaticOperator
{
public:
    CUDA_DECORATOR Exp() : StaticOperator(StaticOperator::Exp)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return exp(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return expf(l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Exp>(*this);
    }
};

class Pow : public StaticOperator
{
public:
    CUDA_DECORATOR Pow() : StaticOperator(StaticOperator::Pow), _value(0.0)
    {

    }

    CUDA_DECORATOR Pow(double v) : StaticOperator(StaticOperator::Pow), _value(v)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return pow(l, _value);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return powf(l, static_cast<float>(_value));
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Pow>(*this);
    }

public:
    double _value;
};

class Abs : public StaticOperator
{
public:
    CUDA_DECORATOR Abs() : StaticOperator(StaticOperator::Abs)
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
        return fabs(l);
    }

    CUDA_DECORATOR size_t operator()(const size_t& l) const
    {
        return l;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Abs>(*this);
    }
};

class Sqrt : public StaticOperator
{
public:
    CUDA_DECORATOR Sqrt() : StaticOperator(StaticOperator::Sqrt)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return sqrt(l);
    }

    CUDA_DECORATOR float operator()(const float& l) const
    {
        return sqrtf(l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Sqrt>(*this);
    }
};

class Multiply : public StaticOperator
{
public:
    CUDA_DECORATOR Multiply() : StaticOperator(StaticOperator::Multiply), _value(0.0)
    {

    }

    CUDA_DECORATOR Multiply(double d) : StaticOperator(StaticOperator::Multiply), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Multiply>(*this);
    }

private:
    double _value;

};

class Divide : public StaticOperator
{
public:
    CUDA_DECORATOR Divide() : StaticOperator(StaticOperator::Divide), _value(0.0)
    {

    }

    CUDA_DECORATOR Divide(double d) : StaticOperator(StaticOperator::Divide), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Divide>(*this);
    }

private:
    double _value;

};

class Sigmoid : public StaticOperator
{
public:
    CUDA_DECORATOR Sigmoid() : StaticOperator(StaticOperator::Sigmoid)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        if(l < -50.0) return -50.0;
        if(l >  50.0) return  50.0;

        return T(1.0) / (T(1.0) + T(exp(-l)));
    }

    CUDA_DECORATOR size_t operator()(const size_t& l) const
    {
        return 1.0 / (1.0 + exp(-(double)l));
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Sigmoid>(*this);
    }

};

class SigmoidDerivative : public StaticOperator
{
public:
    CUDA_DECORATOR SigmoidDerivative() : StaticOperator(StaticOperator::SigmoidDerivative)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return l * (1.0 - l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<SigmoidDerivative>(*this);
    }

};

class RectifiedLinear : public StaticOperator
{
public:
    CUDA_DECORATOR RectifiedLinear() : StaticOperator(StaticOperator::RectifiedLinear)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return (l > 20.0) ? 20.0 : ((l < 0.0) ? 0.0 : l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<RectifiedLinear>(*this);
    }

};

class RectifiedLinearDerivative : public StaticOperator
{
public:
    CUDA_DECORATOR RectifiedLinearDerivative()
    : StaticOperator(StaticOperator::RectifiedLinearDerivative)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return (l >= 20.0) ? 0.0 : ((l <= 0.0) ? 0.0 : 1.0);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<RectifiedLinearDerivative>(*this);
    }

};

class Tanh : public StaticOperator
{
public:
    CUDA_DECORATOR Tanh() : StaticOperator(StaticOperator::Tanh)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return std::tanh(l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Tanh>(*this);
    }

};

class TanhDerivative : public StaticOperator
{
public:
    CUDA_DECORATOR TanhDerivative() : StaticOperator(StaticOperator::TanhDerivative)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return 1.0 - (l * l);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<TanhDerivative>(*this);
    }

};

class Negate : public StaticOperator
{
public:
    CUDA_DECORATOR Negate() : StaticOperator(StaticOperator::Negate)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return -l;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Negate>(*this);
    }

};

class Maximum : public StaticOperator
{
public:
    CUDA_DECORATOR Maximum() : StaticOperator(StaticOperator::Maximum), _value(0.0)
    {

    }

    CUDA_DECORATOR Maximum(double d) : StaticOperator(StaticOperator::Maximum), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return parallel::max(l, r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return parallel::max(T(_value), r);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Maximum>(*this);
    }

private:
    double _value;

};

class Minimum : public StaticOperator
{
public:
    CUDA_DECORATOR Minimum() : StaticOperator(StaticOperator::Minimum), _value(0.0)
    {

    }

    CUDA_DECORATOR Minimum(double d) : StaticOperator(StaticOperator::Minimum), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return parallel::min(l, r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return parallel::min(T(_value), r);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Minimum>(*this);
    }

private:
    double _value;

};

class Equal : public StaticOperator
{
public:
    CUDA_DECORATOR Equal() : StaticOperator(StaticOperator::Equal), _value(0.0)
    {

    }

    CUDA_DECORATOR Equal(double d) : StaticOperator(StaticOperator::Equal), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Equal>(*this);
    }

private:
    double _value;

};

class LessThan : public StaticOperator
{
public:
    CUDA_DECORATOR LessThan() : StaticOperator(StaticOperator::LessThan), _value(0.0)
    {

    }

    CUDA_DECORATOR LessThan(double d) : StaticOperator(StaticOperator::LessThan), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<LessThan>(*this);
    }

private:
    double _value;

};

class NotEqual : public StaticOperator
{
public:
    CUDA_DECORATOR NotEqual() : StaticOperator(StaticOperator::NotEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR NotEqual(double d) : StaticOperator(StaticOperator::NotEqual), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<NotEqual>(*this);
    }

private:
    double _value;

};

class GreaterThanOrEqual : public StaticOperator
{
public:
    CUDA_DECORATOR GreaterThanOrEqual()
    : StaticOperator(StaticOperator::GreaterThanOrEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR GreaterThanOrEqual(double d)
    : StaticOperator(StaticOperator::GreaterThanOrEqual), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<GreaterThanOrEqual>(*this);
    }

private:
    double _value;

};

class GreaterThan : public StaticOperator
{
public:
    CUDA_DECORATOR GreaterThan() : StaticOperator(StaticOperator::GreaterThan), _value(0.0)
    {

    }

    CUDA_DECORATOR GreaterThan(double d) : StaticOperator(StaticOperator::GreaterThan), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return T(l > r);
    }

    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(r > _value);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<GreaterThan>(*this);
    }

private:
    double _value;

};

class LessThanOrEqual : public StaticOperator
{
public:
    CUDA_DECORATOR LessThanOrEqual()
    : StaticOperator(StaticOperator::LessThanOrEqual), _value(0.0)
    {

    }

    CUDA_DECORATOR LessThanOrEqual(double d)
    : StaticOperator(StaticOperator::LessThanOrEqual), _value(d)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<LessThanOrEqual>(*this);
    }

private:
    double _value;

};

class Fill : public StaticOperator
{
public:
    CUDA_DECORATOR Fill(double d = 0.0) : StaticOperator(StaticOperator::Fill), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return _value;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Fill>(*this);
    }

private:
    double _value;

};

class Square : public StaticOperator
{
public:
    CUDA_DECORATOR Square() : StaticOperator(StaticOperator::Square)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r * r;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Square>(*this);
    }

};

class SquareAndScale : public StaticOperator
{
public:
    CUDA_DECORATOR SquareAndScale(double d = 1.0)
    : StaticOperator(StaticOperator::SquareAndScale), _value(d)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return r * r * _value;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<SquareAndScale>(*this);
    }

private:
    double _value;

};

class Inverse : public StaticOperator
{
public:
    CUDA_DECORATOR Inverse() : StaticOperator(StaticOperator::Inverse)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& r) const
    {
        return T(1.0) / r;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Inverse>(*this);
    }

};

class CopyRight : public StaticOperator
{
public:
    CUDA_DECORATOR CopyRight() : StaticOperator(StaticOperator::CopyRight)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l, const T& r) const
    {
        return r;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<CopyRight>(*this);
    }

};

class Nop : public StaticOperator
{
public:
    CUDA_DECORATOR Nop() : StaticOperator(StaticOperator::Nop)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return l;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Nop>(*this);
    }

};

class NopDerivative : public StaticOperator
{
public:
    CUDA_DECORATOR NopDerivative() : StaticOperator(StaticOperator::NopDerivative)
    {

    }

public:
    template<typename T>
    CUDA_DECORATOR T operator()(const T& l) const
    {
        return 1;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<NopDerivative>(*this);
    }

};

class Cos : public StaticOperator
{
public:
    CUDA_DECORATOR Cos() : StaticOperator(StaticOperator::Cos) {}

public:
    CUDA_DECORATOR float operator() (float operand) const
    {
        return std::cos(operand);
    }

    CUDA_DECORATOR double operator() (double operand) const
    {
        return std::cos(operand);
    }

    CUDA_DECORATOR size_t operator() (size_t operand) const
    {
        return std::cos(operand);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Cos>(*this);
    }
};

typedef std::tuple<Add, Subtract, Multiply, Divide, Log, Exp, Pow, Abs, Sqrt, Tanh, TanhDerivative,
                   RectifiedLinear, RectifiedLinearDerivative, Sigmoid, SigmoidDerivative, Negate,
                   Maximum, Minimum, Equal, LessThan, NotEqual, Fill, Square, SquareAndScale,
                   Inverse, Nop, NopDerivative, Cos, GreaterThanOrEqual> AllOperators;

typedef std::tuple<Add, Subtract, Multiply, Divide, Maximum, Minimum,
                   Equal, LessThan, NotEqual, CopyRight, GreaterThanOrEqual> AllBinaryOperators;

typedef std::tuple<Add, Subtract, Multiply, Divide, Log, Exp, Pow, Abs, Sqrt, Tanh, TanhDerivative,
                   RectifiedLinear, RectifiedLinearDerivative, Sigmoid, SigmoidDerivative, Negate,
                   Maximum, Minimum, Equal, LessThan, NotEqual, GreaterThan, Fill, Square,
                   SquareAndScale, Inverse, Nop, NopDerivative, Cos> AllUnaryOperators;

typedef std::tuple<Equal, NotEqual, LessThan, GreaterThan, GreaterThanOrEqual>
                   AllComparisonOperators;

}
}


