
// Lucius Includes
#include <lucius/matrix/interface/AdjacentElementOperations.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/GenericOperators.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/Metaprogramming.h>

namespace lucius
{
namespace matrix
{
namespace detail
{


template<typename NativeType, typename ActualOperation>
class ApplyToAdjacentElementsLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t i = threadGroup.id(); i < elements; i += threadGroup.size())
        {
            auto position = linearToDimension(i, outputView.size());

            auto leftPosition  = position;
            auto rightPosition = position;

            NativeType leftValue = initialValue;

            size_t index = leftPosition[dimensionToApplyTo];

            if(0 != index)
            {
                leftPosition[dimensionToApplyTo] -= 1;
                leftValue = inputView(leftPosition);
            }

            NativeType rightValue = inputView(rightPosition);

            outputView(position) = nativeOperation(leftValue, rightValue);
        }
    }

public:
    MatrixView<NativeType>      outputView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    size_t dimensionToApplyTo;

public:
    ActualOperation nativeOperation;

public:
    double initialValue;

};

template <typename ActualOperation, typename ActualPrecision>
void applyToAdjacentElements(Matrix& output, const Matrix& input,
    size_t dimensionToApplyTo, const StaticOperator& op, double initialValue,
    const std::tuple<ActualPrecision>& p)
{
    typedef typename ActualPrecision::type NativeType;

    assert(ActualPrecision() == output.precision());
    assert(ActualPrecision() == input.precision());
    assert(input.size() == output.size());

    auto nativeOperation = static_cast<const ActualOperation&>(op);

    size_t elements = input.size().product();

    MatrixView<NativeType>      outputView(output);
    ConstMatrixView<NativeType> inputView(input);

    auto lambda = ApplyToAdjacentElementsLambda<NativeType, ActualOperation>{outputView,
        inputView, elements, dimensionToApplyTo, nativeOperation, initialValue};

    parallel::multiBulkSynchronousParallel(lambda);
}


template <typename ActualOperation, typename PossiblePrecisions>
void applyToAdjacentElements(Matrix& output, const Matrix& input,
    size_t dimensionToApplyTo, const StaticOperator& op, double initialValue,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;
    if(output.precision() == PossiblePrecisionType())
    {
        applyToAdjacentElements<ActualOperation>(output, input, dimensionToApplyTo, op,
            initialValue, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;
        applyToAdjacentElements<ActualOperation>(output, input, dimensionToApplyTo, op,
            initialValue, RemainingPrecisions());
    }

}

template <typename PossibleOperation>
void applyToAdjacentElements(Matrix& output, const Matrix& input,
    size_t dimensionToApplyTo, const StaticOperator& op, double initialValue,
    const std::tuple<PossibleOperation>& p)
{
    assert(PossibleOperation() == op);
    applyToAdjacentElements<PossibleOperation>(output, input, dimensionToApplyTo, op,
        initialValue, AllPrecisions());
}

template <typename PossibleOperations>
void applyToAdjacentElements(Matrix& output, const Matrix& input,
    size_t dimensionToApplyTo, const StaticOperator& op, double initialValue,
    const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(op == PossibleOperationType())
    {
        applyToAdjacentElements(output, input, dimensionToApplyTo, op, initialValue,
            std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        applyToAdjacentElements(output, input, dimensionToApplyTo, op, initialValue,
            RemainingOperations());
    }
}

}

Matrix applyToAdjacentElements(const Matrix& input, size_t dimensionToApplyTo,
    const StaticOperator& op, double initialValue)
{
    Matrix result(input.size(), input.precision());

    applyToAdjacentElements(result, input, dimensionToApplyTo, op, initialValue);

    return result;
}

void applyToAdjacentElements(Matrix& output, const Matrix& input,
    size_t dimensionToApplyTo, const StaticOperator& op, double initialValue)
{
    detail::applyToAdjacentElements(output, input, dimensionToApplyTo, op,
        initialValue, AllBinaryOperators());
}

}
}



