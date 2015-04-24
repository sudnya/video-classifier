
// Minerva Includes
#include <minerva/matrix/interface/MatrixOperations.h>
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/CopyOperations.h>
#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixView.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/parallel/interface/MultiBulkSynchronousParallel.h>

#include <minerva/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <tuple>

namespace minerva
{
namespace matrix
{

namespace detail
{

template<typename OperationType, typename T>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right,
    const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    assert(result.isContiguous() && left.isContiguous() && right.isContiguous()); // TODO: handle complex strides

    auto rawResult = static_cast<NativeType*>(result.data());
    auto rawLeft   = static_cast<const NativeType*>(left.data());
    auto rawRight  = static_cast<const NativeType*>(right.data());

    size_t elements = result.elements();

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            rawResult[i] = nativeOperation(rawLeft[i], rawRight[i]);
        }
    });
}

template<typename OperationType, typename PossiblePrecisions>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op,
    const Precision& precision, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        applyOverPrecisions<OperationType>(result, left, right, op, precision,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        applyOverPrecisions<OperationType>(result, left, right, op, precision, RemainingPrecisions());
    }
}


template<typename T>
void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op,
    const Precision& precision, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    applyOverPrecisions<PossibleOperationType, AllPrecisions>(result, left, right, op, precision, AllPrecisions());
}

template<typename PossibleOperations>
void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right,
    const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        applyOverOperations(result, left, right, op, precision, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        applyOverOperations(result, left, right, op, precision, RemainingOperations());
    }
}

void applyOverOperations(Matrix& result, const Matrix& left, const Matrix& right,
    const Operation& op, const Precision& precision)
{
    applyOverOperations<AllBinaryOperations>(result, left, right, op, precision, AllBinaryOperations());
}

}

void apply(Matrix& result, const Matrix& left, const Matrix& right, const Operation& op)
{
    auto precision = left.precision();

    detail::applyOverOperations(result, left, right, op, precision);
}

Matrix apply(const Matrix& left, const Matrix& right, const Operation& op)
{
    assert(left.size() == right.size());
    assert(left.precision() == right.precision());

    Matrix temp(left.size(), left.precision());

    apply(temp, left, right, op);

    return temp;
}

namespace detail
{

template<typename OperationType, typename T>
void applyOverPrecisions(Matrix& result, const Matrix& input,
    const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    assert(result.isContiguous() && input.isContiguous()); // TODO: handle complex strides

    auto rawResult = static_cast<NativeType*>(result.data());
    auto rawInput  = static_cast<const NativeType*>(input.data());

    size_t elements = result.elements();

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            rawResult[i] = nativeOperation(rawInput[i]);
        }
    });
}

template<typename OperationType, typename PossiblePrecisions>
void applyOverPrecisions(Matrix& result, const Matrix& input, const Operation& op,
    const Precision& precision, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        applyOverPrecisions<OperationType>(result, input, op, precision,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        applyOverPrecisions<OperationType>(result, input, op, precision, RemainingPrecisions());
    }
}


template<typename T>
void applyOverOperations(Matrix& result, const Matrix& input, const Operation& op,
    const Precision& precision, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    applyOverPrecisions<PossibleOperationType, AllPrecisions>(result, input, op, precision, AllPrecisions());
}

template<typename PossibleOperations>
void applyOverOperations(Matrix& result, const Matrix& input,
    const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        applyOverOperations(result, input, op, precision, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        applyOverOperations(result, input, op, precision, RemainingOperations());
    }
}

void applyOverOperations(Matrix& result, const Matrix& input,
    const Operation& op, const Precision& precision)
{
    applyOverOperations<AllUnaryOperations>(result, input, op, precision, AllUnaryOperations());
}

}

void apply(Matrix& result, const Matrix& input, const Operation& op)
{
    detail::applyOverOperations(result, input, op, input.precision());
}

Matrix apply(const Matrix& input, const Operation& op)
{
    Matrix result(input.size(), input.precision());

    apply(result, input, op);

    return result;
}

namespace detail
{

template <typename ActualOperation, typename ActualPrecision>
void reduce(Matrix& result, const Matrix& input, const Dimension& unsortedDimensions, const Operation& op, const std::tuple<ActualPrecision>& p)
{
    typedef typename ActualPrecision::type NativeType;

    Dimension dimensions = unsortedDimensions;

    std::sort(dimensions.begin(), dimensions.end());

    assert(ActualPrecision()  == result.precision());
    assert(result.precision() == input.precision());
    assert(result.size()      == removeDimensions(input.size(), dimensions));

    size_t elements = result.elements();

    auto nativeOperation = static_cast<const ActualOperation&>(op);

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            auto resultIndex = linearToDimension(i, resultView.size());

            // find the start of the input slice
            auto inputBegin = selectNamedDimensions(dimensions, resultIndex, zeros(inputView.size()));

            // find the end of the input slice
            auto inputEnd = selectNamedDimensions(dimensions, resultIndex + ones(inputBegin.size()), inputView.size());

            auto inputSlice = slice(inputView, inputBegin, inputEnd);

            // find the total size of the slice
            size_t sliceSize = inputSlice.elements();

            // iterate over i linearly from 0 to size
            auto resultValue = inputSlice(zeros(inputSlice.size()));

            for(size_t inputLinearIndex = 1; inputLinearIndex < sliceSize; ++inputLinearIndex)
            {
                // get index for i in the input's space
                auto inputIndex = linearToDimension(inputLinearIndex, inputSlice.size());

                // apply operator to resultValue, input[index]
                resultValue = nativeOperation(resultValue, inputSlice(inputIndex));
            }

            // save the result
            resultView(resultIndex) = resultValue;
        }
    });

}

template <typename ActualOperation, typename PossiblePrecisions>
void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
    if(result.precision() == PossiblePrecisionType())
    {
        reduce<ActualOperation>(result, input, dimensions, op, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        reduce<ActualOperation>(result, input, dimensions, op, RemainingPrecisions());
    }
}

template <typename PossibleOperation>
void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op, const std::tuple<PossibleOperation>& p)
{
    assert(PossibleOperation() == op);
    reduce<PossibleOperation>(result, input, dimensions, op, AllPrecisions());
}

template <typename PossibleOperations>
void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op, const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;
    if(op == PossibleOperationType())
    {
        reduce(result, input, dimensions, op, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        reduce(result, input, dimensions, op, RemainingOperations());
    }

}

}

void reduce(Matrix& result, const Matrix& input, const Dimension& dimensions, const Operation& op)
{
    detail::reduce(result, input, dimensions, op, AllBinaryOperations());
}

Matrix reduce(const Matrix& input, const Dimension& dimensions, const Operation& op)
{
    Matrix result(removeDimensions(input.size(), dimensions), input.precision());

    reduce(result, input, dimensions, op);

    return result;
}

namespace detail
{

static Dimension fillOutDimension(const Dimension& d, const Dimension& leftSize, const Dimension& rightSize)
{
    if (d.size() != 0)
    {
        return d;
    }
    Dimension retVal;
    for (auto i = leftSize.begin(), j = rightSize.begin(); i != leftSize.end(); ++i)
    {
        if ((j != rightSize.end()) && (*i == *j))
        {
            ++j;
            continue;
        }

        retVal.push_back(std::distance(leftSize.begin(), i));
    }

    return retVal;
}

template <typename ActualOperation, typename ActualPrecision>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op,
    const std::tuple<ActualPrecision>& p)
{
    auto dimension = fillOutDimension(d, left.size(), right.size());
    typedef typename ActualPrecision::type NativeType;

    assert(ActualPrecision()  == result.precision());
    assert(result.precision() == left.precision());
    assert(result.precision() == right.precision());
    assert(result.size()      == left.size());

    size_t elements = result.elements();

    auto nativeOperation = static_cast<const ActualOperation&>(op);

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> leftView(left);
    ConstMatrixView<NativeType> rightView(right);

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            auto fullDimension    = linearToDimension(i, resultView.size());
            auto reducedDimension = removeDimensions(fullDimension, dimension);

            resultView(fullDimension) = nativeOperation(leftView(fullDimension), rightView(reducedDimension));
        }
    });
}

template <typename ActualOperation, typename PossiblePrecisions>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
    if(result.precision() == PossiblePrecisionType())
    {
        broadcast<ActualOperation>(result, left, right, d, op, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        broadcast<ActualOperation>(result, left, right, d, op, RemainingPrecisions());
    }
}

template <typename PossibleOperation>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op,
    const std::tuple<PossibleOperation>& p)
{
    assert(PossibleOperation() == op);
    broadcast<PossibleOperation>(result, left, right, d, op, AllPrecisions());
}

template <typename PossibleOperations>
void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op,
    const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;
    if(op == PossibleOperationType())
    {
        broadcast(result, left, right, d, op, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        broadcast(result, left, right, d, op, RemainingOperations());
    }

}

}

void broadcast(Matrix& result, const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op)
{
    detail::broadcast(result, left, right, d, op, AllBinaryOperations());
}

Matrix broadcast(const Matrix& left, const Matrix& right, const Dimension& d, const Operation& op)
{
    Matrix retVal(left.size(), left.precision());
    broadcast(retVal, left, right, d, op);
    return retVal;
}

void zeros(Matrix& result)
{
	apply(result, result, Fill(0.0));
}

Matrix zeros(const Dimension& size, const Precision& precision)
{
    Matrix retVal(size, precision);

    zeros(retVal);

    return retVal;
}

void ones(Matrix& result)
{
	apply(result, result, Fill(1.0));
}

Matrix ones(const Dimension& size, const Precision& precision)
{
    Matrix retVal(size, precision);

    ones(retVal);

    return retVal;
}

}
}


