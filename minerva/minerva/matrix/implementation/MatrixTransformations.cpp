
// Minerva Includes
#include <minerva/matrix/interface/MatrixTransformations.h>
#include <minerva/matrix/interface/CopyOperations.h>

// Standard Library Includes
#include <set>

namespace minerva
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

Dimension linearStride(const Dimension& size)
{
    Dimension stride;

    size_t step = 1;

    for (auto sizeStep : size)
    {
        stride.push_back(step);
        step *= sizeStep;
    }

    return stride;
}

Dimension zeros(const Dimension& size)
{
    Dimension result;

    for(size_t i = 0, arity = size.size(); i < arity; ++i)
    {
        result.push_back(0);
    }

    return result;
}

Dimension ones(const Dimension& size)
{
    Dimension result;

    for(size_t i = 0, arity = size.size(); i < arity; ++i)
    {
        result.push_back(1);
    }

    return result;
}

Dimension removeDimensions(const Dimension& base, const Dimension& toRemove)
{
    if(toRemove.size() == 0)
    {
        return Dimension({1});
    }

    std::set<size_t> removed;

    for(auto i : toRemove)
    {
        removed.insert(i);
    }

    Dimension result;

    for(size_t i = 0; i < base.size(); ++i)
    {
        if(removed.count(i) == 0)
        {
            result.push_back(base[i]);
        }
    }

    return result;
}

Dimension intersection(const Dimension& left, const Dimension& right)
{
    size_t totalDimensions = std::min(left.size(), right.size());

    Dimension result;

    for(size_t i = 0; i < totalDimensions; ++i)
    {
        result.push_back(std::min(left[i], right[i]));
    }

    return result;
}

size_t dotProduct(const Dimension& left, const Dimension& right)
{
    assert(left.size() == right.size());

    size_t product = 0;

    for(auto i = left.begin(), j = right.begin(); i != left.end(); ++i, ++j)
    {
        product += *i * *j;
    }

    return product;
}

Dimension linearToDimension(size_t linearIndex, const Dimension& size)
{
    Dimension result;

    for(auto dimensionSize : size)
    {
        result.push_back(linearIndex % dimensionSize);

        linearIndex /= dimensionSize;
    }

    return result;
}

Dimension selectNamedDimensions(const Dimension& selectedDimensions, const Dimension& left, const Dimension& right)
{
    Dimension result;

    if(selectedDimensions.size() == 0)
    {
        return right;
    }

    size_t selectedDimensionIndex = 0;
    size_t leftIndex = 0;

    for(size_t rightIndex = 0; rightIndex != right.size(); ++rightIndex)
    {
        if(selectedDimensionIndex < selectedDimensions.size() && selectedDimensions[selectedDimensionIndex] == rightIndex)
        {
            result.push_back(right[rightIndex]);
            ++selectedDimensionIndex;
        }
        else
        {
            result.push_back(left[leftIndex]);
            ++leftIndex;
        }
    }

    return result;
}

static size_t getOffset(const Dimension& stride, const Dimension& position)
{
    size_t offset = 0;
    size_t arity = std::min(stride.size(), position.size());

    for(auto i = 0; i < arity; ++i)
    {
        offset += stride[i] * position[i];
    }

    return offset;
}

void* getAddress(const Dimension& stride, const Dimension& position, void* data, const Precision& precision)
{
    size_t offset = getOffset(stride, position);

    uint8_t* address = static_cast<uint8_t*>(data);

    return address + precision.size() * offset;
}

const void* getAddress(const Dimension& stride, const Dimension& position, const void* data, const Precision& precision)
{
    size_t offset = getOffset(stride, position);

    const uint8_t* address = static_cast<const uint8_t*>(data);

    return address + precision.size() * offset;
}

}
}


