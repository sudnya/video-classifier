
// Lucius Includes
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/CopyOperations.h>

// Standard Library Includes
#include <set>
#include <vector>

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

typedef std::vector<size_t>   InputMap;
typedef std::vector<InputMap> OutputToInputMap;

static OutputToInputMap createDimensionMap(const Dimension& outputSize,
    const Dimension& inputSize, const Dimension& inputStride)
{
    size_t inputIndex = 0;
    size_t inputRemainder = inputSize.front();

    OutputToInputMap map;

    for(size_t outputIndex = 0; outputIndex < outputSize.size(); ++outputIndex)
    {
        InputMap localMap;

        size_t outputDimSize = outputSize[outputIndex];

        while(outputDimSize > inputRemainder)
        {
            if(inputRemainder > 1)
            {
                localMap.push_back(inputIndex);
            }

            ++inputIndex;
            inputRemainder *= inputSize[inputIndex];
        }

        localMap.push_back(inputIndex);
        inputRemainder /= outputDimSize;

        map.push_back(localMap);
    }

    return map;
}

static bool checkThatMergedDimensionsAreContiguous(const OutputToInputMap& map,
    const Dimension& inputSize, const Dimension& inputStride)
{
    for(size_t i = 0; i < map.size(); ++i)
    {
        auto& ids = map[i];

        if(ids.size() == 1)
        {
            continue;
        }

        for(size_t j = 1; j < ids.size(); ++j)
        {
            if(inputStride[ids[j]] != inputSize[ids[j - 1]] * inputStride[ids[j - 1]])
            {
                return false;
            }
        }
    }

    return true;
}

static Dimension convertMapToStrides(const OutputToInputMap& map, const Dimension& outputSize,
    const Dimension& inputStride)
{
    size_t lastDimension = inputStride.size();
    size_t strideMultiplier = 1;
    size_t strideBase = 1;

    Dimension outputStride;

    for(size_t i = 0; i < map.size(); ++i)
    {
        auto& inputIndices = map[i];

        if(lastDimension == inputIndices[0])
        {
            strideMultiplier *= outputSize[i - 1];
        }
        else
        {
            strideMultiplier = 1;
            strideBase = inputStride[inputIndices[0]];
        }

        outputStride.push_back(strideBase * strideMultiplier);
        lastDimension = inputIndices.back();
    }

    return outputStride;
}

/*
    Compute a stride for a reshaped array.  Handle the case when the array is strided.
*/
static Dimension fillInStride(const Dimension& newSize, const Dimension& inputStride,
    const Dimension& inputSize)
{
    auto dimensionMap = createDimensionMap(newSize, inputSize, inputStride);

    checkThatMergedDimensionsAreContiguous(dimensionMap, inputSize, inputStride);

    return convertMapToStrides(dimensionMap, newSize, inputStride);
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


