
// Lucius Includes
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <tuple>

namespace lucius
{
namespace matrix
{

namespace detail
{

template<typename NativeResultType, typename NativeInputType>
class CopyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            rawResult[i] = rawInput[i];
        }
    }

public:
          NativeResultType* rawResult;
    const NativeInputType*  rawInput;

public:
    size_t elements;

};

template<typename NativeResultType, typename NativeInputType>
class NoncontiguousCopyLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto dimension      = linearToDimension(i, resultView.size());
            auto inputDimension = linearToDimension(i,  inputView.size());

            resultView(dimension) = inputView(inputDimension);
        }
    }

public:
    MatrixView<NativeResultType>     resultView;
    ConstMatrixView<NativeInputType> inputView;

public:
    size_t elements;
};

template<typename ResultPrecision, typename InputPrecision>
void copy(Matrix& result, const Matrix& input)
{
    typedef typename ResultPrecision::type NativeResultType;
    typedef typename InputPrecision::type  NativeInputType;

    size_t elements = result.elements();

    if(result.isContiguous() && input.isContiguous())
    {
        auto rawResult = static_cast<NativeResultType*>(result.data());
        auto rawInput  = static_cast<const NativeInputType*>(input.data());

        auto lambda = CopyLambda<NativeResultType, NativeInputType>
            {rawResult, rawInput, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        MatrixView<NativeResultType>     resultView(result);
        ConstMatrixView<NativeInputType> inputView(input);

        auto lambda = NoncontiguousCopyLambda<NativeResultType, NativeInputType>
            {resultView, inputView, elements};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template<typename ResultPrecision, typename T>
void copyOverInputPrecisions(Matrix& result, const Matrix& input,
    const std::tuple<T>& inputPrecisions)
{
    typedef T PossibleInputPrecision;

    assert(PossibleInputPrecision() == input.precision());

    copy<ResultPrecision, PossibleInputPrecision>(result, input);
}

template<typename ResultPrecision, typename InputPrecisions>
void copyOverInputPrecisions(Matrix& result, const Matrix& input,
    const InputPrecisions& inputPrecisions)
{
    typedef typename std::tuple_element<0, InputPrecisions>::type PossibleInputPrecision;

    if(input.precision() == PossibleInputPrecision())
    {
        copy<ResultPrecision, PossibleInputPrecision>(result, input);
    }
    else
    {
        typedef typename util::RemoveFirstType<InputPrecisions>::type RemainingPrecisions;

        copyOverInputPrecisions<ResultPrecision>(result, input, RemainingPrecisions());
    }

}

template<typename T, typename InputPrecisions>
void copyOverPrecisions(Matrix& result, const std::tuple<T>& resultPrecisions,
    const Matrix& input, const InputPrecisions& inputPrecisions)
{
    typedef T PossibleResultPrecision;

    assert(PossibleResultPrecision() == result.precision());

    copyOverInputPrecisions<PossibleResultPrecision>(result, input, inputPrecisions);
}

template<typename ResultPrecisions, typename InputPrecisions>
void copyOverPrecisions(Matrix& result, const ResultPrecisions& resultPrecisions,
    const Matrix& input, const InputPrecisions& inputPrecisions)
{
    typedef typename std::tuple_element<0, ResultPrecisions>::type PossibleResultPrecision;

    if(result.precision() == PossibleResultPrecision())
    {
        copyOverInputPrecisions<PossibleResultPrecision>(result, input, inputPrecisions);
    }
    else
    {
        typedef typename util::RemoveFirstType<ResultPrecisions>::type RemainingPrecisions;

        copyOverPrecisions(result, RemainingPrecisions(), input, inputPrecisions);
    }
}

void copyOverPrecisions(Matrix& result, const Matrix& input)
{
    copyOverPrecisions(result, AllPrecisions(), input, AllPrecisions());
}

}

void copy(Matrix& result, const Matrix& input)
{
    detail::copyOverPrecisions(result, input);
}

Matrix copy(const Matrix& input)
{
    Matrix result(input.size(), input.precision());

    copy(result, input);

    return result;
}

Matrix copy(const Matrix& input, const Precision& precision)
{
    Matrix result(input.size(), precision);

    copy(result, input);

    return result;
}

namespace detail
{

template<typename NativeType, typename OperationType>
class GatherLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            size_t newIndex = nativeOperation(i);

            resultBase[i] = inputBase[newIndex];
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
class NoncontiguousGatherLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto dimension = linearToDimension(i, resultView.size());
            auto inputDimension = linearToDimension(nativeOperation(i), inputView.size());

            resultView(dimension) = inputView(inputDimension);
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
void gatherOverPrecisions(Matrix& result, const Matrix& input,
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

        auto lambda = GatherLambda<NativeType, OperationType>
            {resultBase, inputBase, elements, nativeOperation};

        parallel::multiBulkSynchronousParallel(lambda);
    }
    else
    {
        MatrixView<NativeType>      resultView(result);
        ConstMatrixView<NativeType> inputView(input);

        auto lambda = NoncontiguousGatherLambda<NativeType, OperationType>
            {resultView, inputView, elements, nativeOperation};

        parallel::multiBulkSynchronousParallel(lambda);
    }
}

template<typename OperationType, typename PossiblePrecisions>
void gatherOverPrecisions(Matrix& result, const Matrix& input, const Operation& op,
    const Precision& precision, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        gatherOverPrecisions<OperationType>(result, input, op, precision,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        gatherOverPrecisions<OperationType>(result, input, op, precision, RemainingPrecisions());
    }
}


template<typename T>
void gatherOverOperations(Matrix& result, const Matrix& input, const Operation& op,
    const Precision& precision, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    gatherOverPrecisions<PossibleOperationType, AllPrecisions>(result, input, op,
        precision, AllPrecisions());
}

template<typename PossibleOperations>
void gatherOverOperations(Matrix& result, const Matrix& input,
    const Operation& op, const Precision& precision, const PossibleOperations& operations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        gatherOverOperations(result, input, op, precision, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        gatherOverOperations(result, input, op, precision, RemainingOperations());
    }
}

void gatherOverOperations(Matrix& result, const Matrix& input,
    const Operation& op, const Precision& precision)
{
    gatherOverOperations<AllGatherOperations>(result, input, op, precision, AllGatherOperations());
}

}

void gather(Matrix& result, const Matrix& input, const Operation& op)
{
    detail::gatherOverOperations(result, input, op, input.precision());
}

Matrix gather(const Matrix& input, const Operation& op)
{
    Matrix result(input.size(), input.precision());

    gather(result, input, op);

    return result;
}

void permuteDimensions(Matrix& result, const Matrix& input, const Dimension& newOrder)
{
    gather(result, input, PermuteDimensionGather(input.stride(), result.size(), newOrder));
}

Matrix permuteDimensions(const Matrix& input, const Dimension& newOrder)
{
    Matrix result(selectDimensions(input.size(), newOrder), input.precision());

    permuteDimensions(result, input, newOrder);

    return result;
}

}
}



