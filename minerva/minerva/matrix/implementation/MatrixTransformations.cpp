
// Lucious Includes
#include <lucious/matrix/interface/MatrixTransformations.h>
#include <lucious/matrix/interface/DimensionTransformations.h>
#include <lucious/matrix/interface/CopyOperations.h>

// Standard Library Includes
#include <set>

namespace lucious
{
namespace matrix
{

static Dimension fillInDimension(const Dimension& newSize, const Dimension& inputSize)
{
    if(newSize.size() > inputSize.size())
    {
        assert(newSize.product() == inputSize.product());

        return newSize;
    }

    Dimension size(newSize);

    // fill in remaining non-empty dimensions
    size_t remaining = inputSize.product() / size.product();

    size_t dimension = size.size();

    assert(inputSize.product() % size.product() == 0);

    // TODO: be smarter about the remainder
    for(size_t d = dimension; d < inputSize.size(); ++d)
    {
        if(remaining <= 1)
        {
            break;
        }

        size.push_back(remaining);
        remaining /= remaining;
    }

    assert(size.product() == inputSize.product());

    return size;
}

static Dimension computeSpacing(const Dimension& stride, const Dimension& size)
{
    Dimension spacing;

    size_t linearStep = 1;

    for(size_t i = 0; i < stride.size(); ++i)
    {
        spacing.push_back(stride[i] / linearStep);
        linearStep *= stride[i] * size[i] / linearStep;
    }

    return spacing;
}

static Dimension fillInStride(const Dimension& newSize, const Dimension& inputStride, const Dimension& inputSize)
{
    Dimension inputSpacing = computeSpacing(inputStride, inputSize);

    Dimension newStride = linearStride(newSize);

    // extend the input spacing with 1
    for(size_t i = inputSpacing.size(); i < newStride.size(); ++i)
    {
        inputSpacing.push_back(1);
    }

    // update the stride with the existing spacing
    for(size_t i = 0, spacingMultiplier = 1; i < newStride.size(); ++i)
    {
        spacingMultiplier *= inputSpacing[i];

        inputSpacing[i] *= spacingMultiplier;
    }

    return newStride;
}

Matrix reshape(const Matrix& input, const Dimension& size)
{
    Matrix tempInput(input);

    auto newSize = fillInDimension(size, input.size());

    return Matrix(newSize, fillInStride(newSize, input.stride(), input.size()),
        input.precision(), tempInput.allocation(), tempInput.data());
}

Matrix flatten(const Matrix& matrix)
{
    return reshape(matrix, {matrix.elements()});
}

Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end)
{
    auto size = end - begin;

    Matrix tempInput(input);

    return Matrix(size, input.stride(), input.precision(), tempInput.allocation(), tempInput[begin].address());
}

Matrix slice(const Matrix& input, const Dimension& begin, const Dimension& end, const Dimension& stride)
{
    auto size = (end - begin) / stride;

    Matrix tempInput(input);

    return Matrix(size, input.stride() * stride, input.precision(), tempInput.allocation(), tempInput[begin].address());
}

Matrix resize(const Matrix& input, const Dimension& size)
{
    Matrix result(size, input.precision());

    auto overlap = intersection(size, input.size());

    auto resultSlice = slice(result, zeros(overlap), overlap);
    auto inputSlice  = slice(input,  zeros(overlap), overlap);

    copy(resultSlice, inputSlice);

    return result;
}

}
}


