
// Lucius Includes
#include <lucius/matrix/interface/ReduceByKeyOperations.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/DimensionTransformations.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/MetaProgramming.h>

namespace lucius
{
namespace matrix
{
namespace detail
{

template <typename OperationType, typename NativeType>
class ReduceByKeyLambda
{
public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        for(size_t notReducedElement = threadGroup.id(); notReducedElement < notReducedElements;
            notReducedElement += threadGroup.size())
        {
            Dimension notReducedDimension = linearToDimension(notReducedElement, notReducedSize);

            NativeType outputValue = 0;
            NativeType currentKey  = 0;
            size_t resultIndex     = 0;

            bool shouldInitialize = true;

            for(size_t reducedElement = 0; reducedElement < reducedElements; ++reducedElement)
            {
                Dimension reducedDimension = linearToDimension(reducedElement, reducedSize);

                Dimension completeDimension = mergeNamedDimensions(dimensionsToReduce,
                    reducedDimension, notReducedDimension);

                NativeType nextValue = valuesView(completeDimension);
                NativeType nextKey   = keysView(completeDimension);

                if(shouldInitialize)
                {
                    shouldInitialize = false;

                    outputValue = nextValue;
                    currentKey  = nextKey;
                }
                else if(currentKey == nextKey)
                {
                    outputValue = operation(outputValue, nextValue);
                }
                else
                {
                    saveResult(outputValue, resultIndex, notReducedDimension);
                    ++resultIndex;

                    outputValue = nextValue;
                    currentKey  = nextKey;
                }
            }

            saveResult(outputValue, resultIndex, notReducedDimension);
        }
    }

    CUDA_DECORATOR void saveResult(const NativeType& outputValue, size_t resultIndex,
        const Dimension& notReducedDimension) const
    {
        Dimension resultDimension = linearToDimension(resultIndex, reducedSize);

        Dimension completeResultDimension = mergeNamedDimensions(
            dimensionsToReduce, resultDimension, notReducedDimension);

        resultView(completeResultDimension) = outputValue;
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> keysView;
    ConstMatrixView<NativeType> valuesView;

public:
    size_t notReducedElements;
    size_t reducedElements;

public:
    Dimension dimensionsToReduce;

public:
    Dimension notReducedSize;
    Dimension reducedSize;

public:
    OperationType operation;
};

template <typename OperationType, typename PrecisionType>
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values,
    const Dimension& dimensionsToReduce, const Operation& op,
    const std::tuple<PrecisionType>& precisions)
{
    assert(PrecisionType() == result.precision());
    assert(PrecisionType() ==   keys.precision());
    assert(PrecisionType() == values.precision());

    assert(keys.size()   == values.size());
    assert(result.size() == values.size());

    typedef typename PrecisionType::type NativeType;

    auto& nativeOperation = static_cast<const OperationType&>(op);

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> keysView(keys);
    ConstMatrixView<NativeType> valuesView(values);

    auto notReducedSize = removeDimensions(result.size(), dimensionsToReduce);
    auto reducedSize    = selectDimensions(result.size(), dimensionsToReduce);

    auto lambda = ReduceByKeyLambda<OperationType, NativeType>{resultView, keysView,
        valuesView, notReducedSize.product(), reducedSize.product(), dimensionsToReduce,
        notReducedSize, reducedSize, nativeOperation};

    parallel::multiBulkSynchronousParallel(lambda);
}

template <typename OperationType, typename PossiblePrecisions>
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values,
    const Dimension& dimensionsToReduce, const Operation& op,
    const PossiblePrecisions& precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(result.precision() == PossiblePrecisionType())
    {
        reduceByKey<OperationType>(result, keys, values, dimensionsToReduce,
            op, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        reduceByKey<OperationType>(result, keys, values, dimensionsToReduce,
            op, RemainingPrecisions());
    }
}

template <typename OperationType>
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values,
    const Dimension& dimensionsToReduce, const Operation& op,
    const std::tuple<OperationType>& possibleOperation)
{
    assert(OperationType() == op);

    reduceByKey<OperationType>(result, keys, values, dimensionsToReduce, op, AllPrecisions());
}

template <typename PossibleOperations>
void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values,
    const Dimension& dimensionsToReduce, const Operation& op,
    const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        reduceByKey(result, keys, values, dimensionsToReduce,
            op, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        reduceByKey(result, keys, values, dimensionsToReduce, op, RemainingOperations());
    }
}

} // namespace detail

Matrix reduceByKey(const Matrix& keys, const Matrix& values, const Dimension& dimensionsToReduce,
    const Operation& op)
{
    Matrix result = zeros(values.size(), values.precision());

    reduceByKey(result, keys, values, dimensionsToReduce, op);

    return result;
}

Matrix reduceByKey(const Matrix& keys, const Matrix& values,
    const Operation& op)
{
    return reduceByKey(keys, values, range(values.size()), op);
}

void reduceByKey(Matrix& result, const Matrix& keys, const Matrix& values,
    const Dimension& dimensionsToReduce, const Operation& op)
{
    detail::reduceByKey(result, keys, values, dimensionsToReduce, op, AllBinaryOperations());
}

} // namespace matrix
} // namespace lucius

