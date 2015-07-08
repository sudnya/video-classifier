
// Lucious Includes
#include <lucious/matrix/interface/MatrixOperations.h>
#include <lucious/matrix/interface/MatrixTransformations.h>
#include <lucious/matrix/interface/CopyOperations.h>
#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixView.h>
#include <lucious/matrix/interface/Operation.h>

#include <lucious/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucious/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <tuple>

namespace lucious
{
namespace matrix
{

namespace detail
{

template<typename NativeType, typename OperationType>
class BinaryApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            resultBase[i] = nativeOperation(leftBase[i], rightBase[i]);
        }
    }

public:
          NativeType* resultBase;
    const NativeType* leftBase;
    const NativeType* rightBase;

public:
    OperationType nativeOperation;

public:
    size_t elements;

};

template<typename NativeType, typename OperationType>
class BinaryNoncontiguousApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto fullDimension = linearToDimension(i, resultView.size());

            resultView(fullDimension) = nativeOperation(leftView(fullDimension), rightView(fullDimension));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> leftView;
    ConstMatrixView<NativeType> rightView;

public:
    OperationType nativeOperation;

public:
    size_t elements;

};

template<typename OperationType, typename T>
void applyOverPrecisions(Matrix& result, const Matrix& left, const Matrix& right,
    const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    size_t elements = result.elements();

    if(result.isContiguous() && left.isContiguous() && right.isContiguous())
    {
        auto resultBase = static_cast<NativeType*>(result.data());
        auto leftBase   = static_cast<const NativeType*>(left.data());
        auto rightBase  = static_cast<const NativeType*>(right.data());

        auto lambda = BinaryApplyLambda<NativeType, OperationType>
            {resultBase, leftBase, rightBase, nativeOperation, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> leftView(left);
        ConstMatrixView<NativeType> rightView(right);

        auto lambda = BinaryNoncontiguousApplyLambda<NativeType, OperationType>
            {resultView, leftView, rightView, nativeOperation, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
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

    assert(left.size() == right.size());
    assert(result.size() == right.size());
    assert(left.precision() == right.precision());
    assert(result.precision() == right.precision());

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

template<typename NativeType, typename OperationType>
class UnaryApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            resultBase[i] = nativeOperation(inputBase[i]);
        }
    }

public:
          NativeType* resultBase;
    const NativeType* inputBase;

public:
    size_t elements;

public:
    OperationType nativeOperation;
};

template<typename NativeType, typename OperationType>
class UnaryNoncontiguousApplyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto dimension = linearToDimension(i, resultView.size());

            resultView(dimension) = nativeOperation(inputView(dimension));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    OperationType nativeOperation;
};

template<typename OperationType, typename T>
void applyOverPrecisions(Matrix& result, const Matrix& input,
    const Operation& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    size_t elements = result.elements();

    if(input.isContiguous() && result.isContiguous())
    {
        auto resultBase = static_cast<NativeType*>(result.data());
        auto inputBase  = static_cast<const NativeType*>(input.data());

        auto lambda = UnaryApplyLambda<NativeType, OperationType>
            {resultBase, inputBase, elements, nativeOperation};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {

        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> inputView(input);

        auto lambda = UnaryNoncontiguousApplyLambda<NativeType, OperationType>
            {resultView, inputView, elements, nativeOperation};

        parallel::multiBulkSynchronousParallel(lambda);
    }
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

static const size_t tileSize = 8;

template<typename NativeType, typename ActualOperation>
class ReduceAllDimensionsStepLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            NativeType value = rawInput[i];

            for(size_t inputElement = i + elements; inputElement < inputElements; inputElement += elements)
            {
                value = nativeOperation(value, rawInput[inputElement]);
            }

            rawResult[i] = value;
        }
    }

public:
          NativeType* rawResult;
    const NativeType* rawInput;

public:
    size_t elements;
    size_t inputElements;

public:
    ActualOperation nativeOperation;

};

template <typename ActualOperation, typename ActualPrecision>
void reduceAllDimensionsStep(Matrix& result, const Matrix& input, const ActualOperation& nativeOperation, const ActualPrecision& p)
{
    typedef typename ActualPrecision::type NativeType;

    NativeType*       rawResult = static_cast<NativeType*>(result.data());
    const NativeType* rawInput  = static_cast<const NativeType*>(input.data());

    auto lambda = ReduceAllDimensionsStepLambda<NativeType, ActualOperation>{rawResult, rawInput, result.elements(), input.elements(), nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template <typename ActualOperation, typename ActualPrecision>
void reduceAllDimensions(Matrix& result, const Matrix& input, const ActualOperation& nativeOperation, const ActualPrecision& p)
{
    if(input.elements() < tileSize)
    {
        return reduceAllDimensionsStep(result, input, nativeOperation, p);
    }

    Matrix temporary({(input.elements() + tileSize - 1) / tileSize}, p);

    reduceAllDimensionsStep(temporary, input, nativeOperation, p);

    reduceAllDimensions(result, temporary, nativeOperation, p);
}

template<typename NativeType, typename ActualOperation>
class GenericReduceLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            auto resultIndex = linearToDimension(i, resultView.size());

            // find the start of the input slice
            auto inputBegin = selectNamedDimensions(dimensions, resultIndex, zeros(inputView.size()));

            // find the end of the input slice
            auto inputEnd = selectNamedDimensions(dimensions, resultIndex + ones(resultView.size()), inputView.size());

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

    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    ActualOperation nativeOperation;

public:
    Dimension dimensions;

};

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

    // Reduce down to a single element
    if(elements == 1)
    {
        reduceAllDimensions(result, input, nativeOperation, ActualPrecision());
    }
    else
    {
        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> inputView(input);

        auto lambda = GenericReduceLambda<NativeType, ActualOperation>{resultView, inputView, elements, nativeOperation, dimensions};

        parallel::multiBulkSynchronousParallel(lambda);
    }

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

template <typename NativeType, typename ActualOperation>
class BroadcastLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            auto fullDimension    = linearToDimension(i, resultView.size());
            auto reducedDimension = removeDimensions(fullDimension, dimension);

            resultView(fullDimension) = nativeOperation(leftView(fullDimension), rightView(reducedDimension));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> leftView;
    ConstMatrixView<NativeType> rightView;

public:
    size_t elements;

public:
    ActualOperation nativeOperation;

public:
    Dimension dimension;

};

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

    auto lambda = BroadcastLambda<NativeType, ActualOperation>{resultView, leftView, rightView, elements, nativeOperation, dimension};

    parallel::multiBulkSynchronousParallel(lambda);
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


