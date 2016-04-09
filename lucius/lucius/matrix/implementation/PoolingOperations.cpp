
// Lucius Includes
#include <lucius/matrix/interface/PoolingOperations.h>

#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/CopyOperations.h>
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

static void genericForwardMaxPooling(Matrix& result,
    const Matrix& input, const Dimension& poolingSize)
{
    size_t remainingElements = input.elements() / poolingSize.product();

    auto reshapedInput = reshape(
        gather(input,
            matrix::Pool2DGather(poolingSize[0], poolingSize[1], input.size()[0], input.size()[1])),
        {poolingSize[0], poolingSize[1], remainingElements});

    auto reshapedResult = reshape(result, Dimension(remainingElements));

    reduce(reshapedResult, reshapedInput, {0, 1}, matrix::Maximum());
}

static void cudnnForwardMaxPooling(Matrix& result,
    const Matrix& input, const Dimension& poolingSize)
{
    CudnnPooling2dDescriptor poolingDescriptor(poolingSize[0], poolingSize[1], 0, 0,
        poolingSize[0], poolingSize[1]);

    CudnnTensorDescriptor inputDescriptor(input);

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
}

static Dimension getForwardMaxPoolingSize(const Dimension& inputSize, const Dimension& poolingSize)
{
    auto resultSize = inputSize;
/*
    if(CudnnLibrary::loaded())
    {
        int n = 0;
        int c = 0;
        int h = 0;
        int w = 0;

        CudnnPooling2dDescriptor poolingDescriptor(inputSize[0], inputSize[1], 0, 0,
            poolingSize[0], poolingSize[1]);

        CudnnTensorDescriptor inputDescriptor(inputSize);

        CudnnLibrary::cudnnGetPooling2dForwardOutputDim(
            poolingDescriptor.descriptor(),
            inputDescriptor.descriptor(),
            &n,
            &c,
            &h,
            &w);

        resultSize[0] = w;
        resultSize[1] = h;
        resultSize[2] = c;
        resultSize[3] = n;
    }
    else
    {*/
        resultSize[0] /= poolingSize[0];
        resultSize[1] /= poolingSize[1];

    //}

    return resultSize;
}

Matrix forwardMaxPooling(const Matrix& input, const Dimension& poolingSize)
{
    auto resultSize = getForwardMaxPoolingSize(input.size(), poolingSize);

    Matrix result(resultSize, input.precision());

    forwardMaxPooling(result, input, poolingSize);

    return result;
}

void forwardMaxPooling(Matrix& result, const Matrix& input, const Dimension& poolingSize)
{
    assert(poolingSize.size() == 2);

    // TODO: support non-evenly divisible sizes
    assert(input.size()[0] % poolingSize[0] == 0);
    assert(input.size()[1] % poolingSize[1] == 0);

    assert(result.precision() == input.precision());
    assert(result.size() == getForwardMaxPoolingSize(input.size(), poolingSize));

    if(CudnnLibrary::loaded())
    {
        cudnnForwardMaxPooling(result, input, poolingSize);
        return;
    }

    genericForwardMaxPooling(result, input, poolingSize);
}


static void cudnnBackwardMaxPooling(Matrix& result, const Matrix& inputActivations,
    const Matrix& outputActivations, const Matrix& outputDeltas, const Dimension& poolingSize)
{
    CudnnPooling2dDescriptor poolingDescriptor(poolingSize[0],
        poolingSize[1], 0, 0, poolingSize[0], poolingSize[1]);

    CudnnTensorDescriptor inputDescriptor(inputActivations);
    CudnnTensorDescriptor outputDescriptor(outputActivations);
    CudnnTensorDescriptor deltasDescriptor(outputDeltas);

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
}

static void genericBackwardMaxPooling(Matrix& result, const Matrix& inputActivations,
    const Matrix& outputActivations, const Matrix& outputDeltas, const Dimension& poolingSize)
{
    size_t remainingElements = inputActivations.elements() / poolingSize.product();

    auto reshapedInput = reshape(
        gather(inputActivations, matrix::Pool2DGather(poolingSize[0], poolingSize[1],
            inputActivations.size()[0], inputActivations.size()[1])),
        {poolingSize[0], poolingSize[1], remainingElements});

    auto maxPositions = reduceGetPositions(reshapedInput, {0, 1}, matrix::Maximum());

    auto resultDeltas = broadcast(maxPositions, outputDeltas, {0, 1}, matrix::CopyRight());

    auto reshapedResult = apply(Matrix(resultDeltas), maxPositions, matrix::Multiply());

    gather(result, reshapedResult,
        matrix::Pool2DGatherInverse(poolingSize[0], poolingSize[1],
        inputActivations.size()[0], inputActivations.size()[1]));
}

Matrix backwardMaxPooling(const Matrix& inputActivations, const Matrix& outputActivations,
    const Matrix& outputDeltas, const Dimension& poolingSize)
{
    Matrix result(inputActivations.size(), inputActivations.precision());

    backwardMaxPooling(result, inputActivations, outputActivations, outputDeltas, poolingSize);

    return result;
}

void backwardMaxPooling(Matrix& result, const Matrix& inputActivations,
    const Matrix& outputActivations, const Matrix& outputDeltas, const Dimension& poolingSize)
{
    assert(result.size() == inputActivations.size());

    assert(outputActivations.size() ==
        getForwardMaxPoolingSize(inputActivations.size(), poolingSize));
    assert(outputActivations.size() == outputDeltas.size());

    assert(result.precision() == inputActivations.precision());
    assert(outputActivations.precision() == inputActivations.precision());
    assert(outputDeltas.precision() == inputActivations.precision());

    if(CudnnLibrary::loaded())
    {
        cudnnBackwardMaxPooling(result, inputActivations, outputActivations, outputDeltas,
            poolingSize);
        return;
    }

    genericBackwardMaxPooling(result, inputActivations, outputActivations, outputDeltas,
        poolingSize);
}

}
}



