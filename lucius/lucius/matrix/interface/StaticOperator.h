/*  \file   StaticOperator.h
    \date   Sunday August 11, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the StaticOperator classes.
*/

#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>

// Standard Library Includes
#include <memory>

namespace lucius
{
namespace matrix
{

/*! \brief A class for specifying basic matrix operations. */
class StaticOperator
{
public:
    enum Type
    {
        // generic
        Add,
        Subtract,
        Multiply,
        Divide,
        Log,
        Exp,
        Pow,
        Abs,
        Sqrt,
        Sigmoid,
        SigmoidDerivative,
        RectifiedLinear,
        RectifiedLinearDerivative,
        Tanh,
        TanhDerivative,
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
        GreaterThan,
        Fill,
        Square,
        SquareAndScale,
        Inverse,
        CopyRight,
        Nop,
        NopDerivative,
        Cos,

        // gather
        Pool2DGather,
        Pool2DGatherInverse,
        PermuteDimensionGather,
        HanningGather,
        GatherIndexToOneHot,
        MapOutputToIndexDimension,
        MapOutputToMatchingIndexDimension
    };

public:
    CUDA_DECORATOR StaticOperator(Type t) : _type(t) {}

public:
    CUDA_DECORATOR bool operator==(const StaticOperator&) const;

public:
    virtual std::unique_ptr<StaticOperator> clone() const = 0;

private:
    Type _type;

};

}
}

