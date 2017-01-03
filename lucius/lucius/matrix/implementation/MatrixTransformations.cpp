
// Lucius Includes
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/CopyOperations.h>

// Standard Library Includes
#include <set>

namespace lucius
{
namespace matrix
{

static Dimension fillInDimension(const Dimension& newSize, const Dimension& inputSize)
{
    if(newSize.product() == inputSize.product())
    {
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

static Dimension compressSpacing(const Dimension& uncompressedInputSpacing,
    const Dimension& newSize, const Dimension& inputSize)
{
    Dimension inputSpacing;

    size_t currentSpacing = 1;
    size_t currentSize    = 1;
    size_t currentIndex   = 0;

    for(size_t i = 0; i < inputSize.size(); ++i)
    {
        currentSpacing *= uncompressedInputSpacing[i];
        currentSize    *= inputSize[i];

        if(currentIndex >= newSize.size())
        {
            inputSpacing.push_back(currentSpacing);
        }
        else if(newSize[currentIndex] == currentSize)
        {
            inputSpacing.push_back(currentSpacing);

            currentSize    = 1;
            currentSpacing = 1;
            ++currentIndex;
        }
    }

    return inputSpacing;
}

static Dimension fillInStride(const Dimension& newSize,
    const Dimension& inputStride, const Dimension& inputSize)
{
    Dimension uncompressedInputSpacing = computeSpacing(inputStride, inputSize);

    Dimension inputSpacing = compressSpacing(uncompressedInputSpacing, newSize, inputSize);

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

        newStride[i] *= spacingMultiplier;
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
    assert(begin.size() == input.size().size());
    assert(end.size()   == input.size().size());

    assert(begin <= input.size());
    assert(end   <= input.size());

    auto size = end - begin;

    Matrix tempInput(input);

    return Matrix(size, input.stride(), input.precision(),
        tempInput.allocation(), tempInput[begin].address());
}

Matrix slice(const Matrix& input, const Dimension& begin,
    const Dimension& end, const Dimension& stride)
{
    assert(begin.size() == input.size().size());
    assert(end.size()   == input.size().size());

    assert(begin <= input.size());
    assert(end   <= input.size());

    auto size = (end - begin) / stride;

    Matrix tempInput(input);

    return Matrix(size, input.stride() * stride, input.precision(),
        tempInput.allocation(), tempInput[begin].address());
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

Matrix concatenate(const Matrix& left, const Matrix& right, size_t dimension)
{
    auto size = left.size();

    size[dimension] += right.size()[dimension];

    Matrix result(size, left.precision());

    auto leftStart  = zeros(size);
    auto rightStart = zeros(size);

    rightStart[dimension] = left.size()[dimension];

    auto leftSlice  = slice(result, leftStart,  left.size());
    auto rightSlice = slice(result, rightStart, result.size());

    copy(leftSlice,  left);
    copy(rightSlice, right);

    return result;
}

}
}


