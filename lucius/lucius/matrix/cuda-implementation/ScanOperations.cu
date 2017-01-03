
// Lucius Includes
#include <lucius/matrix/interface/ScanOperations.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/Metaprogramming.h>

namespace lucius
{
namespace matrix
{
namespace detail
{

template <typename OperationType, typename NativeType, bool isInclusive>
class NoncontiguousScanLambda
{
public:
    void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            auto notReducedPosition = linearToDimension(element, notReducedDimensions);

            auto position = mergeNamedDimensions({dimensionToReduce}, {0}, notReducedPosition);

            NativeType counter = initialValue;

            for(size_t reducedDimensionPosition = 0;
                reducedDimensionPosition < reducedDimensionSize; ++reducedDimensionPosition)
            {
                auto currentValue = inputView(position);

                if(isInclusive)
                {
                    counter += currentValue;
                }

                resultView(position) = counter;

                if(!isInclusive)
                {
                    counter += currentValue;
                }

                position[dimensionToReduce]++;
            }
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    OperationType nativeOperation;

public:
    NativeType initialValue;

public:
    Dimension notReducedDimensions;
    size_t    elements;
    size_t    dimensionToReduce;
    size_t    reducedDimensionSize;

};

template <bool isInclusive, typename OperationType, typename PrecisionType>
void scan(Matrix& result, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue, const std::tuple<PrecisionType>& precisions)
{
    assert(PrecisionType() == result.precision());
    assert(PrecisionType() == input.precision());
    assert(result.size() == input.size());

    typedef typename PrecisionType::type NativeType;

    auto& nativeOperation = static_cast<const OperationType&>(op);

    Dimension notReducedDimensions = removeDimensions(result.size(), {dimensionToReduce});
    size_t elements = notReducedDimensions.product();
    size_t reducedDimensionSize = result.size()[dimensionToReduce];

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    auto lambda = NoncontiguousScanLambda<OperationType, NativeType, isInclusive>{resultView,
        inputView, nativeOperation, static_cast<NativeType>(initialValue), notReducedDimensions,
        elements, dimensionToReduce, reducedDimensionSize};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<bool isInclusive, typename OperationType, typename PossiblePrecisions>
void scan(Matrix& result, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue, const PossiblePrecisions& precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(result.precision() == PossiblePrecisionType())
    {
        scan<isInclusive, OperationType>(result, input, dimensionToReduce, op, initialValue,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        scan<isInclusive, OperationType>(result, input, dimensionToReduce, op, initialValue,
            RemainingPrecisions());
    }
}

template <bool isInclusive, typename OperationType>
void scan(Matrix& result, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue, const std::tuple<OperationType>& operations)
{
    assert(op == OperationType());

    scan<isInclusive, OperationType>(result, input, dimensionToReduce, op, initialValue,
        AllPrecisions());
}

template <bool isInclusive, typename PossibleOperations>
void scan(Matrix& result, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue, const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        scan<isInclusive>(result, input, dimensionToReduce, op, initialValue,
            std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        scan<isInclusive>(result, input, dimensionToReduce, op, initialValue,
            RemainingOperations());
    }
}

} // namespace detail

void inclusiveScan(Matrix& output, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue)
{
    detail::scan<true>(output, input, dimensionToReduce, op, initialValue, AllBinaryOperations());
}

Matrix inclusiveScan(const Matrix& input, size_t dimensionToReduce, const Operation& op,
    double initialValue)
{
    Matrix result(input.size(), input.precision());

    inclusiveScan(result, input, dimensionToReduce, op, initialValue);

    return result;
}

void exclusiveScan(Matrix& output, const Matrix& input, size_t dimensionToReduce,
    const Operation& op, double initialValue)
{
    detail::scan<false>(output, input, dimensionToReduce, op, initialValue, AllBinaryOperations());
}

Matrix exclusiveScan(const Matrix& input, size_t dimensionToReduce, const Operation& op,
    double initialValue)
{
    Matrix result(input.size(), input.precision());

    exclusiveScan(result, input, dimensionToReduce, op, initialValue);

    return result;
}

}
}


