/*  \file   ScalarOperations.cpp
    \date   April 17, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for common operations on scalars.
*/

// Lucius Includes
#include <lucius/matrix/interface/ScalarOperations.h>

#include <lucius/matrix/interface/Scalar.h>
#include <lucius/matrix/interface/GenericOperators.h>

#include <lucius/util/interface/Metaprogramming.h>

namespace lucius
{
namespace matrix
{
namespace detail
{

template <typename ActualOperator, typename PossiblePrecision>
void apply(Scalar& result, const Scalar& left, const Scalar& right, const StaticOperator& op,
    const std::tuple<PossiblePrecision>& p)
{
    assert(left.getPrecision() == PossiblePrecision());
    assert(right.getPrecision() == PossiblePrecision());

    using NativeType = typename PossiblePrecision::type;

    auto& nativeOperator = static_cast<const ActualOperator&>(op);

    auto leftValue  = left.get<NativeType>();
    auto rightValue = right.get<NativeType>();

    result.set(nativeOperator(leftValue, rightValue));
}

template <typename ActualOperator, typename PossiblePrecisions>
void apply(Scalar& result, const Scalar& left, const Scalar& right, const StaticOperator& op,
    const PossiblePrecisions& p)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
    if(left.getPrecision() == PossiblePrecisionType())
    {
        apply<ActualOperator>(result, left, right, op,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        apply<ActualOperator>(result, left, right, op, RemainingPrecisions());
    }

}

template <typename PossibleOperator>
void apply(Scalar& result, const Scalar& left, const Scalar& right, const StaticOperator& op,
    const std::tuple<PossibleOperator>& p)
{
    assert(PossibleOperator() == op);

    apply<PossibleOperator>(result, left, right, op, AllPrecisions());
}

template <typename PossibleOperators>
void apply(Scalar& result, const Scalar& left, const Scalar& right, const StaticOperator& op,
    const PossibleOperators& p)
{
    typedef typename std::tuple_element<0, PossibleOperators>::type PossibleOperatorType;

    if(op == PossibleOperatorType())
    {
        apply(result, left, right, op, std::tuple<PossibleOperatorType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperators>::type RemainingOperators;

        apply(result, left, right, op, RemainingOperators());
    }
}

} // namespace detail

void apply(Scalar& result, const Scalar& left, const Scalar& right, const StaticOperator& op)
{
    detail::apply(result, left, right, op, AllBinaryOperators());
}

Scalar apply(const Scalar& left, const Scalar& right, const StaticOperator& op)
{
    Scalar result;

    apply(result, left, right, op);

    return result;
}

namespace detail
{

template <typename ActualOperator, typename PossiblePrecision>
void apply(Scalar& result, const Scalar& left, const StaticOperator& op,
    const std::tuple<PossiblePrecision>& p)
{
    assert(left.getPrecision() == PossiblePrecision());

    using NativeType = typename PossiblePrecision::type;

    auto& nativeOperator = static_cast<const ActualOperator&>(op);

    auto leftValue = left.get<NativeType>();

    result.set(nativeOperator(leftValue));
}

template <typename ActualOperator, typename PossiblePrecisions>
void apply(Scalar& result, const Scalar& left,  const StaticOperator& op,
    const PossiblePrecisions& p)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
    if(left.getPrecision() == PossiblePrecisionType())
    {
        apply<ActualOperator>(result, left, op,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        apply<ActualOperator>(result, left, op, RemainingPrecisions());
    }

}

template <typename PossibleOperator>
void apply(Scalar& result, const Scalar& left, const StaticOperator& op,
    const std::tuple<PossibleOperator>& p)
{
    assert(PossibleOperator() == op);

    apply<PossibleOperator>(result, left, op, AllPrecisions());
}

template <typename PossibleOperators>
void apply(Scalar& result, const Scalar& left, const StaticOperator& op,
    const PossibleOperators& p)
{
    typedef typename std::tuple_element<0, PossibleOperators>::type PossibleOperatorType;

    if(op == PossibleOperatorType())
    {
        apply(result, left, op, std::tuple<PossibleOperatorType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperators>::type RemainingOperators;

        apply(result, left, op, RemainingOperators());
    }
}

} // namespace detail

void apply(Scalar& result, const Scalar& input, const StaticOperator& op)
{
    detail::apply(result, input, op, AllUnaryOperators());
}

Scalar apply(const Scalar& input, const StaticOperator& op)
{
    Scalar result;

    apply(result, input, op);

    return result;
}

} // namespace matrix
} // namespace lucius



