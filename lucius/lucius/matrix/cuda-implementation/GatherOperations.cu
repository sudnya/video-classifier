
// Lucius Includes
#include <lucius/matrix/interface/GatherOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/GatherOperators.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/Metaprogramming.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <tuple>

namespace lucius
{
namespace matrix
{

namespace detail
{

template<typename NativeType, typename OperationType>
class NoncontiguousGatherLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto resultDimension = linearToDimension(i, resultView.size());

            resultView(resultDimension) = nativeOperation.runOperator(
                resultDimension, ConstMatrixView<NativeType>(resultView), inputView);
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
    const StaticOperator& op, const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    size_t elements = result.elements();

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    auto lambda = NoncontiguousGatherLambda<NativeType, OperationType>
        {resultView, inputView, elements, nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<typename OperationType, typename PossiblePrecisions>
void gatherOverPrecisions(Matrix& result, const Matrix& input, const StaticOperator& op,
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
void gatherOverOperations(Matrix& result, const Matrix& input, const StaticOperator& op,
    const Precision& precision, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    gatherOverPrecisions<PossibleOperationType, AllPrecisions>(result, input, op,
        precision, AllPrecisions());
}

template<typename PossibleOperations>
void gatherOverOperations(Matrix& result, const Matrix& input,
    const StaticOperator& op, const Precision& precision, const PossibleOperations& operations)
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
    const StaticOperator& op, const Precision& precision)
{
    gatherOverOperations<AllGatherOperators>(result, input, op, precision, AllGatherOperators());
}

} // namespace detail

void gather(Matrix& result, const Matrix& input, const StaticOperator& op)
{
    detail::gatherOverOperations(result, input, op, input.precision());
}

Matrix gather(const Matrix& input, const StaticOperator& op)
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

namespace detail
{

template<typename NativeType, typename OperationType>
class NoncontiguousIndirectGatherLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto resultDimension = linearToDimension(i, resultView.size());

            resultView(resultDimension) = nativeOperation.runOperator(
                resultDimension, ConstMatrixView<NativeType>(resultView), inputView, indexView);
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;
    ConstMatrixView<NativeType> indexView;

public:
    size_t elements;

public:
    OperationType nativeOperation;
};

template<typename OperationType, typename T>
void indirectGatherOverPrecisions(Matrix& result, const Matrix& input, const Matrix& indices,
    const StaticOperator& op, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(result.precision() == PrecisionPrimitive());
    assert(input.precision() == PrecisionPrimitive());
    assert(indices.precision() == PrecisionPrimitive());

    auto nativeOperation = static_cast<const OperationType&>(op);

    size_t elements = result.elements();

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);
    ConstMatrixView<NativeType> indexView(indices);

    auto lambda = NoncontiguousIndirectGatherLambda<NativeType, OperationType>
        {resultView, inputView, indexView, elements, nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<typename OperationType, typename PossiblePrecisions>
void indirectGatherOverPrecisions(Matrix& result, const Matrix& input, const Matrix& indices,
    const StaticOperator& op, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(result.precision() == PossiblePrecisionType())
    {
        indirectGatherOverPrecisions<OperationType>(result, input, indices, op,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        indirectGatherOverPrecisions<OperationType>(result, input, indices, op,
            RemainingPrecisions());
    }
}

template<typename T>
void indirectGatherOverOperations(Matrix& result, const Matrix& input, const Matrix& indices,
    const StaticOperator& op, const std::tuple<T>& operations)
{
    typedef T PossibleOperationType;

    assert(op == PossibleOperationType());

    indirectGatherOverPrecisions<PossibleOperationType>(result, input, indices, op,
        AllPrecisions());
}

template<typename PossibleOperations>
void indirectGatherOverOperations(Matrix& result, const Matrix& input, const Matrix& indices,
    const StaticOperator& op, const PossibleOperations& operations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        indirectGatherOverOperations(result, input, indices, op,
            std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        indirectGatherOverOperations(result, input, indices, op, RemainingOperations());
    }
}

} // namespace detail

void indirectGather(Matrix& result, const Matrix& input, const Matrix& indices,
    const StaticOperator& mapper)
{
    detail::indirectGatherOverOperations(result, input, indices, mapper,
        AllIndirectGatherOperators());
}

}
}




