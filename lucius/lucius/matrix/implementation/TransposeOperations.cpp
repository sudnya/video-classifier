
// Lucius Includes
#include <lucius/matrix/interface/TransposeOperations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/Dimension.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/Metaprogramming.h>

namespace lucius
{
namespace matrix
{

Matrix transpose(const Matrix& input)
{
    auto newSize = input.size();

    if(newSize.size() > 1)
    {
        std::swap(newSize[0], newSize[1]);
    }

    Matrix result(newSize, input.precision());

    transpose(result, input);

    return result;
}

namespace detail
{

template<typename T>
void transposeOverPrecisions(Matrix& result, const Matrix& input,
    const Precision& precision, std::tuple<T> precisions)
{
    typedef T PrecisionPrimitive;
    typedef typename PrecisionPrimitive::type NativeType;

    assert(precision == PrecisionPrimitive());
    assert(result.precision() == input.precision());

    assert(result.size().size() > 1);
    assert(input.size().size() > 1);

    assert(result.elements() == input.elements());

    assert(result.size()[0] == input.size()[1]);
    assert(result.size()[1] == input.size()[0]);

    size_t elements = result.elements();

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t i = threadGroup.id(), step = threadGroup.size(); i < elements; i += step)
        {
            auto dimension = linearToDimension(i, resultView.size());
            auto inputDimension = dimension;

            size_t temp = inputDimension[0];
            inputDimension[0] = inputDimension[1];
            inputDimension[1] = temp;

            resultView(dimension) = inputView(inputDimension);
        }
    });
}

template<typename PossiblePrecisions>
void transposeOverPrecisions(Matrix& result, const Matrix& input,
    const Precision& precision, PossiblePrecisions precisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(precision == PossiblePrecisionType())
    {
        transposeOverPrecisions(result, input, precision,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        transposeOverPrecisions(result, input, precision, RemainingPrecisions());
    }
}

}

void transpose(Matrix& output, const Matrix& input)
{
    detail::transposeOverPrecisions(output, input, input.precision(), AllPrecisions());
}

}
}



