
// Lucius Includes
#include <lucius/matrix/interface/PoolingOperations.h>

#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/CudnnLibrary.h>
#include <lucius/matrix/interface/CudnnDescriptors.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>

namespace lucius
{
namespace matrix
{

static Matrix genericForwardMaxPooling(const Matrix& input, const Dimension& poolingSize)
{
    size_t remainingElements = input.elements() / poolingSize.product();

    auto reshapedInput = reshape(input, {poolingSize[0], poolingSize[1], remainingElements});

    auto resultSize = input.size();

    resultSize[0] /= poolingSize[0];
    resultSize[1] /= poolingSize[1];

    auto result = reduce(reshapedInput, {0, 1}, matrix::Maximum());

    return reshape(result, resultSize);
}

static Matrix cudnnForwardMaxPooling(const Matrix& input, const Dimension& poolingSize)
{
    CudnnPooling2dDescriptor poolingDescriptor(input.size()[0], input.size()[1], 0, 0,
        poolingSize[0], poolingSize[1]);

    CudnnTensorDescriptor inputDescriptor(input);

    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;

    CudnnLibrary::cudnnGetPooling2dForwardOutputDim(
        poolingDescriptor.descriptor(),
        inputDescriptor.descriptor(),
        &n,
        &c,
        &h,
        &w);

    Matrix result(Dimension(w, h, c, n), input.precision());

    CudnnTensorDescriptor resultDescriptor(result);

    CudnnScalar alpha(1.0, input.precision());
    CudnnScalar beta( 0.0, input.precision());

    CudnnLibrary::cudnnPoolingForward(
        poolingDescriptor.descriptor(),
        alpha.data(),
        inputDescriptor.descriptor(),
        inputDescriptor.data(),
        beta.data(),
        resultDescriptor.descriptor(),
        resultDescriptor.data()
        );

    return result;
}

Matrix forwardMaxPooling(const Matrix& input, const Dimension& poolingSize)
{
    assert(poolingSize.size() == 2);

    // TODO: support non-evenly divisible sizes
    assert(input.size()[0] % poolingSize[0] == 0);
    assert(input.size()[1] % poolingSize[1] == 0);

    if(CudnnLibrary::loaded())
    {
        return cudnnForwardMaxPooling(input, poolingSize);
    }

    return genericForwardMaxPooling(input, poolingSize);
}


static Matrix cudnnBackwardMaxPooling(const Matrix& inputActivations,
    const Matrix& outputActivations, const Matrix& outputDeltas, const Dimension& poolingSize)
{
    CudnnPooling2dDescriptor poolingDescriptor(inputActivations.size()[0],
        inputActivations.size()[1], 0, 0, poolingSize[0], poolingSize[1]);

    CudnnTensorDescriptor inputDescriptor(inputActivations);
    CudnnTensorDescriptor outputDescriptor(outputActivations);
    CudnnTensorDescriptor deltasDescriptor(outputDeltas);

    Matrix result(inputActivations.size(), inputActivations.precision());

    CudnnTensorDescriptor resultDescriptor(result);

    CudnnScalar alpha(1.0, inputActivations.precision());
    CudnnScalar beta( 0.0, inputActivations.precision());

    CudnnLibrary::cudnnPoolingBackward(
        poolingDescriptor.descriptor(),
        alpha.data(),
        outputDescriptor.descriptor(),
        outputDescriptor.data(),
        deltasDescriptor.descriptor(),
        deltasDescriptor.data(),
        inputDescriptor.descriptor(),
        inputDescriptor.data(),
        beta.data(),
        resultDescriptor.descriptor(),
        resultDescriptor.data()
        );

    return result;

}

static Matrix genericBackwardMaxPooling(const Matrix& inputActivations,
    const Matrix& outputActivations, const Matrix& outputDeltas, const Dimension& poolingSize)
{
    size_t remainingElements = inputActivations.elements() / poolingSize.product();

    auto reshapedInput = reshape(inputActivations, {poolingSize[0], poolingSize[1],
        remainingElements});

    Matrix result(inputActivations.size(), inputActivations.precision());

    auto maxPositions = reduceGetPositions(reshapedInput, {0, 1}, matrix::Maximum());

    broadcast(result, maxPositions, outputDeltas, {0, 1}, matrix::CopyRight());

    apply(result, result, maxPositions, matrix::Multiply());

    return result;
}

Matrix backwardMaxPooling(const Matrix& inputActivations, const Matrix& outputActivations,
    const Matrix& outputDeltas, const Dimension& poolingSize)
{
    if(CudnnLibrary::loaded())
    {
        return cudnnBackwardMaxPooling(inputActivations, outputActivations, outputDeltas,
            poolingSize);
    }

    return genericBackwardMaxPooling(inputActivations, outputActivations, outputDeltas,
        poolingSize);
}

}
}



