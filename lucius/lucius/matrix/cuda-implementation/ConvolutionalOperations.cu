
// Lucius Includes
#include <lucius/matrix/interface/ConvolutionalOperations.h>
#include <lucius/matrix/interface/CudnnLibrary.h>

#include <lucius/matrix/interface/BlasOperations.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/MatrixTransformations.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>

#include <lucius/util/interface/Metaprogramming.h>
#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <vector>

namespace lucius
{
namespace matrix
{

namespace
{

size_t computeOutputSize(size_t inputSize, size_t filterSize, size_t stride, size_t pad)
{
    return (inputSize + 2 * pad - filterSize + stride) / stride;
}

}

Dimension forwardConvolutionOutputSize(const Dimension& inputSize, const Dimension& filterSize,
    const Dimension& filterStride, const Dimension& padding)
{
    Dimension outputSize(
            computeOutputSize(inputSize[0], filterSize[0], filterStride[0], padding[0]),
            computeOutputSize(inputSize[1], filterSize[1], filterStride[1], padding[1]),
            filterSize[3],
            inputSize[3]);

    if(inputSize.size() > 4)
    {
        outputSize.push_back(inputSize[4]);
    }

    return outputSize;
}

namespace
{

CudnnLibrary::cudnnDataType_t getCudnnDataType(const Precision& precision)
{
    if(precision == DoublePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_DOUBLE;
    }

    assert(precision == SinglePrecision());

    return CudnnLibrary::CUDNN_DATA_FLOAT;
}

class CudnnFilterDescriptor
{
public:
    CudnnFilterDescriptor(const Matrix& filter)
    : _filter(filter)
    {
        CudnnLibrary::cudnnCreateFilterDescriptor(&_descriptor);

        CudnnLibrary::cudnnSetFilter4dDescriptor(_descriptor,
            getCudnnDataType(filter.precision()),
            filter.size()[3],
            filter.size()[2],
            filter.size()[1],
            filter.size()[0]);
    }

    ~CudnnFilterDescriptor()
    {
        CudnnLibrary::cudnnDestroyFilterDescriptor(_descriptor);
    }

public:
    CudnnLibrary::cudnnFilterDescriptor_t descriptor() const
    {
        return _descriptor;
    }

public:
    void* data()
    {
        return _filter.data();
    }

private:
    CudnnLibrary::cudnnFilterDescriptor_t _descriptor;

private:
    Matrix _filter;

};

class CudnnTensorDescriptor
{
public:
    CudnnTensorDescriptor(const Matrix& tensor)
    : _tensor(tensor)
    {
        CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

        CudnnLibrary::cudnnSetTensor4dDescriptor(_descriptor,
            CudnnLibrary::CUDNN_TENSOR_NCHW,
            getCudnnDataType(tensor.precision()), // image data type
            _tensor.size()[3],        // number of inputs (batch size)
            _tensor.size()[2],        // number of input feature maps
            _tensor.size()[1],        // height of input section
            _tensor.size()[0]         // width of input section
        );

    }

    ~CudnnTensorDescriptor()
    {
        CudnnLibrary::cudnnDestroyTensorDescriptor(_descriptor);
    }

public:
    CudnnLibrary::cudnnTensorDescriptor_t descriptor() const
    {
        return _descriptor;
    }

    void* data()
    {
        return _tensor.data();
    }

private:
    CudnnLibrary::cudnnTensorDescriptor_t _descriptor;

private:
    Matrix _tensor;

};

class CudnnScalar
{
public:
    CudnnScalar(double value, const Precision& p)
    : _doubleValue(value), _floatValue(value), _precision(p)
    {

    }

    void* data()
    {
        if(_precision == SinglePrecision())
        {
            return &_floatValue;
        }
        else
        {
            return &_doubleValue;
        }
    }

private:
    double _doubleValue;
    float  _floatValue;

private:
    Precision _precision;

};

class CudnnForwardWorkspace
{
public:
    CudnnForwardWorkspace(const CudnnTensorDescriptor& source, const CudnnFilterDescriptor& filter,
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor, const CudnnTensorDescriptor& result)
    {
        CudnnLibrary::cudnnGetConvolutionForwardAlgorithm(
            source.descriptor(),
            filter.descriptor(),
            convolutionDescriptor,
            result.descriptor(),
            CudnnLibrary::CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, // TODO: make this a knob
            0,
            &_algorithm);

        size_t bytes = 0;

        CudnnLibrary::cudnnGetConvolutionForwardWorkspaceSize(
            source.descriptor(),
            filter.descriptor(),
            convolutionDescriptor,
            result.descriptor(),
            _algorithm,
            &bytes);

        _data = std::make_unique<Allocation>(bytes);
    }

public:
    CudnnLibrary::cudnnConvolutionFwdAlgo_t algorithm()
    {
        return _algorithm;
    }

    void* data()
    {
        return _data->data();
    }

    size_t size() const
    {
        return _data->size();
    }

private:
    CudnnLibrary::cudnnConvolutionFwdAlgo_t _algorithm;

private:
    std::unique_ptr<Allocation> _data;
};

template<typename NativeType>
class ForwardReshapeInputLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t miniBatch   = (column / (p * q));
            size_t featureMap  = (row    / (r * s));
            size_t tileOffset  = (row    % (r * s));
            size_t tileRow     = tileOffset % s;
            size_t tileColumn  = tileOffset / s;

            size_t inputTileOffset = column % (p * q);
            size_t inputRow        = ((inputTileOffset % q) * v + tileRow);
            size_t inputColumn     = ((inputTileOffset / q) * u + tileColumn);

            if(inputRow < padWidth || (inputRow >= (padWidth + w)))
            {
                resultView(Dimension(row, column)) = 0.0;
                continue;
            }

            inputRow -= padWidth;

            if(inputColumn < padHeight || (inputColumn >= (padHeight + h)))
            {
                resultView(Dimension(row, column)) = 0.0;
                continue;
            }

            inputColumn -= padHeight;

            resultView(Dimension(row, column)) =
                inputView(Dimension(inputRow, inputColumn, featureMap, miniBatch));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t w;
    size_t h;

public:
    size_t q;
    size_t p;

public:
    size_t s;
    size_t r;

public:
    size_t v;
    size_t u;

public:
    size_t padWidth;
    size_t padHeight;

};

template<typename PrecisionType>
Matrix gatherForwardConvolutionInputOverPrecisions(const Matrix& input, const Matrix& filter,
    const Dimension& filterStride, const Dimension& padding,
    const std::tuple<PrecisionType>& )
{
    assert(input.precision() == PrecisionType());
    assert(input.precision() == filter.precision());

    size_t w = input.size()[0];
    size_t h = input.size()[1];

    size_t padWidth  = padding[0];
    size_t padHeight = padding[1];

    size_t q = computeOutputSize(input.size()[0], filter.size()[0], filterStride[0], padding[0]);
    size_t p = computeOutputSize(input.size()[1], filter.size()[1], filterStride[1], padding[1]);

    size_t s = filter.size()[0];
    size_t r = filter.size()[1];

    size_t v = filterStride[0];
    size_t u = filterStride[1];

    size_t rows    = input.size()[2] * filter.size()[0] * filter.size()[1];
    size_t columns = input.size()[3] * p * q;

    Matrix result({rows, columns}, input.precision());

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    size_t elements = rows * columns;

    auto lambda = ForwardReshapeInputLambda<NativeType>{resultView, inputView, elements,
        rows, w, h, q, p, s, r, v, u, padWidth, padHeight};

    parallel::multiBulkSynchronousParallel(lambda);

    return result;
}

template<typename PossiblePrecisions>
Matrix gatherForwardConvolutionInputOverPrecisions(const Matrix& input, const Matrix& filter,
    const Dimension& stride, const Dimension& padding,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(input.precision() == PossiblePrecisionType())
    {
        return gatherForwardConvolutionInputOverPrecisions(input, filter, stride,
            padding, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return gatherForwardConvolutionInputOverPrecisions(input, filter, stride,
            padding, RemainingPrecisions());
    }

}

Matrix gatherForwardConvolutionInput(const Matrix& input, const Matrix& filter,
    const Dimension& stride, const Dimension& padding)
{
    return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, padding, AllPrecisions());
}

template<typename NativeType>
class ForwardReshapeFilterLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t k = row;

            size_t c = column / (filterR * filterS);
            size_t filterTile = column % (filterR * filterS);

            size_t s = filterTile / filterR;
            size_t r = filterTile % filterR;

            resultView(Dimension(row, column)) =
                filterView(Dimension(filterR - r - 1, filterS - s - 1, c, k));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> filterView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t filterR;
    size_t filterS;

};

template<typename PrecisionType>
Matrix gatherForwardConvolutionFilterOverPrecisions(const Matrix& filter,
    const std::tuple<PrecisionType>& )
{
    assert(filter.precision() == PrecisionType());

    size_t rows    = filter.size()[3];
    size_t columns = filter.size()[2] * filter.size()[0] * filter.size()[1];

    size_t filterR = filter.size()[0];
    size_t filterS = filter.size()[1];

    Matrix result({rows, columns}, filter.precision());

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> filterView(filter);

    size_t elements = rows * columns;

    auto lambda = ForwardReshapeFilterLambda<NativeType>{resultView, filterView,
        elements, rows, filterR, filterS};

    parallel::multiBulkSynchronousParallel(lambda);

    return result;
}

template<typename PossiblePrecisions>
Matrix gatherForwardConvolutionFilterOverPrecisions(const Matrix& filter,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(filter.precision() == PossiblePrecisionType())
    {
        return gatherForwardConvolutionFilterOverPrecisions(filter, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return gatherForwardConvolutionFilterOverPrecisions(filter, RemainingPrecisions());
    }
}

Matrix gatherForwardConvolutionFilter(const Matrix& filter)
{
    return gatherForwardConvolutionFilterOverPrecisions(filter, AllPrecisions());
}

template<typename NativeType>
class ForwardReshapeResultLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t w = element % width;
            size_t h = (element / width) % height;
            size_t featureMap = (element / (width * height)) % featureMaps;
            size_t miniBatch  = (element / (width * height * featureMaps));

            size_t row    = featureMap;
            size_t column = w + h * width + miniBatch * width * height;

            resultView(Dimension(w, h, featureMap, miniBatch)) = inputView(Dimension(row, column));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    size_t width;
    size_t height;
    size_t featureMaps;


};

template<typename PrecisionType>
void scatterForwardConvolutionResultOverPrecisions(Matrix& result, const Matrix& input,
    const std::tuple<PrecisionType>& )
{
    assert(input.precision() == PrecisionType());

    size_t miniBatches = result.size()[3];
    size_t featureMaps = result.size()[2];
    size_t height      = result.size()[1];
    size_t width       = result.size()[0];

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    size_t elements = miniBatches * featureMaps * height * width;

    auto lambda = ForwardReshapeResultLambda<NativeType>{resultView, inputView, elements,
        width, height, featureMaps};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<typename PossiblePrecisions>
void scatterForwardConvolutionResultOverPrecisions(Matrix& result, const Matrix& input,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(input.precision() == PossiblePrecisionType())
    {
        return scatterForwardConvolutionResultOverPrecisions(result,
            input, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return scatterForwardConvolutionResultOverPrecisions(result, input, RemainingPrecisions());
    }
}

void scatterForwardConvolutionResult(Matrix& result, const Matrix& reshapedResult)
{
    scatterForwardConvolutionResultOverPrecisions(result, reshapedResult, AllPrecisions());
}

Matrix foldTime(const Matrix& input)
{
    assert(input.size().size() < 6);

    if(input.size().size() == 5)
    {
        auto size = input.size();
        size_t timesteps = size.back();

        size.pop_back();

        size.back() *= timesteps;

        return reshape(input, size);
    }

    return input;
}

void genericForwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter,
    const Dimension& stride, const Dimension& padding)
{
    auto inputFoldedTime  = foldTime(input);
    auto resultFoldedTime = foldTime(result);

    auto reshapedInput  = gatherForwardConvolutionInput(inputFoldedTime, filter, stride, padding);
    auto reshapedFilter = gatherForwardConvolutionFilter(filter);

    auto reshapedResult = gemm(reshapedFilter, reshapedInput);

    scatterForwardConvolutionResult(resultFoldedTime, reshapedResult);
}

}

void forwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter,
    const Dimension& stride, const Dimension& padding)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            padding[1],                // zero-padding height
            padding[0],                // zero-padding width
            stride[1],        // vertical filter stride
            stride[0],        // horizontal filter stride
            1,                // upscale the input in x-direction
            1,                // upscale the input in y-direction
            CudnnLibrary::CUDNN_CONVOLUTION // convolution mode
        );

        CudnnFilterDescriptor filterDescriptor(filter);
        CudnnTensorDescriptor inputDescriptor(input);
        CudnnTensorDescriptor resultDescriptor(result);

        CudnnScalar alpha(1.0, input.precision());
        CudnnScalar beta( 0.0, input.precision());

        CudnnForwardWorkspace workspace(inputDescriptor, filterDescriptor,
            convolutionDescriptor, resultDescriptor);

        CudnnLibrary::cudnnConvolutionForward(
            alpha.data(),                      //alpha,
            inputDescriptor.descriptor(),      // srcDesc,
            inputDescriptor.data(),            //*srcData,
            filterDescriptor.descriptor(),     // filterDesc,
            filterDescriptor.data(),           //*filterData,
            convolutionDescriptor,             // convDesc,
            workspace.algorithm(),             // algo,
            workspace.data(),                  //*workSpace,
            workspace.size(),                  // workSpaceSizeInBytes,
            beta.data(),                       //*beta,
            resultDescriptor.descriptor(),     // destDesc,
            resultDescriptor.data());          //*destData);

        CudnnLibrary::cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    }
    else
    {
        genericForwardConvolution(result, input, filter, stride, padding);
    }
}

Matrix forwardConvolution(const Matrix& input, const Matrix& filter, const Dimension& stride,
    const Dimension& padding)
{
    auto resultSize = forwardConvolutionOutputSize(input.size(), filter.size(), stride, padding);

    Matrix result(resultSize, input.precision());

    forwardConvolution(result, input, filter, stride, padding);

    return result;
}

namespace
{

size_t invertPadding(size_t inputPadding, size_t filterSize)
{
    assert(inputPadding < filterSize);
    return filterSize - inputPadding - 1;
}

template<typename NativeType>
class ReverseReshapeDeltasInputLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t miniBatch   = (column / (inputW * inputH));
            size_t featureMap  = (row    / (filterW * filterH));

            size_t tileOffset  = (row    % (filterW * filterH));
            size_t tileRow     = tileOffset % filterW;
            size_t tileColumn  = tileOffset / filterW;

            size_t inputTileOffset = column % (inputW * inputH);
            size_t inputTileRow    = (inputTileOffset % inputW);
            size_t inputTileColumn = (inputTileOffset / inputW);

            size_t deltaRow        = (inputTileRow + tileRow);
            size_t deltaColumn     = (inputTileColumn + tileColumn);

            if(deltaRow < padWidth)
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            deltaRow -= padWidth;

            if(deltaRow % strideW != 0)
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            deltaRow /= strideW;

            if(deltaRow >= deltaW)
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            if(deltaColumn < padHeight)
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            deltaColumn -= padHeight;

            if(deltaColumn % strideH != 0)
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            deltaColumn /= strideH;

            if(deltaColumn >= deltaH)
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            resultView(Dimension(row, column)) =
                deltasView(Dimension(deltaRow, deltaColumn, featureMap, miniBatch));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> deltasView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t deltaW;
    size_t deltaH;

public:
    size_t strideW;
    size_t strideH;

public:
    size_t inputW;
    size_t inputH;

public:
    size_t filterW;
    size_t filterH;

public:
    size_t padWidth;
    size_t padHeight;

};

template<typename PrecisionType>
Matrix gatherReverseConvolutionDeltasInput(const Matrix& deltas, const Matrix& filter,
    const Dimension& filterStride, const Dimension& padding, const Dimension& inputSize,
    const PrecisionType& )
{
    typedef typename PrecisionType::type NativeType;

    // zero fill the deltas to full convolution
    size_t deltaW = deltas.size()[0];
    size_t deltaH = deltas.size()[1];

    size_t outputFeatureMaps = deltas.size()[2];
    size_t miniBatches       = deltas.size()[3];

    size_t strideW = filterStride[0];
    size_t strideH = filterStride[1];

    size_t inputW = inputSize[0];
    size_t inputH = inputSize[1];

    size_t filterW = filter.size()[1];
    size_t filterH = filter.size()[0];

    size_t padWidth  = invertPadding(padding[0], filterW);
    size_t padHeight = invertPadding(padding[1], filterH);

    size_t rows    = outputFeatureMaps * filterW * filterH;
    size_t columns = miniBatches * inputW * inputH;

    Matrix result({rows, columns}, deltas.precision());

    zeros(result);

    size_t elements = rows * columns;

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> deltasView(deltas);

    auto lambda = ReverseReshapeDeltasInputLambda<NativeType>{resultView, deltasView,
        elements, rows, deltaW, deltaH, strideW, strideH, inputW, inputH, filterW,
        filterH, padWidth, padHeight};

    parallel::multiBulkSynchronousParallel(lambda);

    return result;
}

template<typename NativeType>
class ReverseDeltasFilterLambda
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t k = row;

            size_t c = column / (filterR * filterS);
            size_t filterTile = column % (filterR * filterS);

            size_t s = filterTile / filterR;
            size_t r = filterTile % filterR;

            resultView(Dimension(row, column)) = filterView(Dimension(r, s, k, c));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> filterView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t filterR;
    size_t filterS;

};

template<typename PrecisionType>
Matrix gatherReverseConvolutionDeltasFilter(const Matrix& filter, const PrecisionType& )
{
    assert(filter.precision() == PrecisionType());

    size_t rows    = filter.size()[2];
    size_t columns = filter.size()[3] * filter.size()[0] * filter.size()[1];

    size_t filterR = filter.size()[0];
    size_t filterS = filter.size()[1];

    Matrix result({rows, columns}, filter.precision());

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> filterView(filter);

    size_t elements = rows * columns;

    auto lambda = ReverseDeltasFilterLambda<NativeType>{resultView, filterView, elements,
        rows, filterR, filterS};

    parallel::multiBulkSynchronousParallel(lambda);

    return result;
}

template<typename NativeType>
class ReverseDeltasReshapeResult
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t w = element % width;
            size_t h = (element / width) % height;
            size_t featureMap = (element / (width * height)) % featureMaps;
            size_t miniBatch  = (element / (width * height * featureMaps));

            size_t row    = featureMap;
            size_t column = w + h * width + miniBatch * width * height;

            resultView(Dimension(w, h, featureMap, miniBatch)) = inputView(Dimension(row, column));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    size_t width;
    size_t height;
    size_t featureMaps;

};

template<typename PrecisionType>
void scatterReverseConvolutionDeltasResult(Matrix& result, const Matrix& input,
    const PrecisionType& )
{
    assert(input.precision() == PrecisionType());

    size_t miniBatches = result.size()[3];
    size_t featureMaps = result.size()[2];
    size_t height      = result.size()[1];
    size_t width       = result.size()[0];

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    size_t elements = miniBatches * featureMaps * height * width;

    auto lambda = ReverseDeltasReshapeResult<NativeType>{resultView, inputView, elements,
        width, height, featureMaps};

    parallel::multiBulkSynchronousParallel(lambda);

}

template<typename PrecisionType>
void genericReverseConvolutionDeltasOverPrecisions(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const Dimension& padding,
    const std::tuple<PrecisionType>& )
{
    assert(filter.precision()       == PrecisionType());
    assert(deltas.precision()       == PrecisionType());
    assert(resultDeltas.precision() == PrecisionType());

    auto deltasFoldedTime       = foldTime(deltas);
    auto resultDeltasFoldedTime = foldTime(resultDeltas);

    // flip the filter
    auto reshapedInputDeltas = gatherReverseConvolutionDeltasInput(deltasFoldedTime, filter,
        stride, padding, resultDeltas.size(), PrecisionType());
    auto reshapedFilter      = gatherReverseConvolutionDeltasFilter(filter, PrecisionType());


    util::log("ConvolutionalOperations") << "           input deltas: " << deltasFoldedTime.toString();
    util::log("ConvolutionalOperations") << "  reshaped input deltas: " << reshapedInputDeltas.toString();


    auto reshapedResultDeltas = gemm(reshapedFilter, reshapedInputDeltas);

    // then multiply like forward convolution
    scatterReverseConvolutionDeltasResult(resultDeltasFoldedTime, reshapedResultDeltas,
        PrecisionType());
}

template<typename PossiblePrecisions>
void genericReverseConvolutionDeltasOverPrecisions(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const Dimension& padding,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(deltas.precision() == PossiblePrecisionType())
    {
        return genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride,
            deltas, padding, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter,
            stride, deltas, padding, RemainingPrecisions());
    }
}

void genericReverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const Dimension& padding)
{
    genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride,
        deltas, padding, AllPrecisions());
}

}

void reverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const Dimension& padding)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            padding[1],       // zero-padding height
            padding[0],       // zero-padding width
            stride[1],        // vertical filter stride
            stride[0],        // horizontal filter stride
            1,                // upscale the input in x-direction
            1,                // upscale the input in y-direction
            CudnnLibrary::CUDNN_CONVOLUTION // convolution mode
        );

        CudnnFilterDescriptor filterDescriptor(filter);
        CudnnTensorDescriptor deltasDescriptor(deltas);
        CudnnTensorDescriptor resultDescriptor(resultDeltas);

        CudnnScalar alpha(1.0, deltas.precision());
        CudnnScalar beta( 0.0, deltas.precision());

        CudnnLibrary::cudnnConvolutionBackwardData(
            alpha.data(),
            filterDescriptor.descriptor(),
            filterDescriptor.data(),
            deltasDescriptor.descriptor(),
            deltasDescriptor.data(),
            convolutionDescriptor,
            beta.data(),
            resultDescriptor.descriptor(),
            resultDescriptor.data());

        CudnnLibrary::cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    }
    else
    {
        genericReverseConvolutionDeltas(resultDeltas, filter, stride, deltas, padding);
    }
}

namespace
{

template<typename NativeType>
class ReverseGradientsReshapeInput
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t miniBatch       = (row    / (outputH * outputW));
            size_t inputFeatureMap = (column / (filterW * filterH));

            size_t tileOffset        = column % (filterW * filterH);
            size_t tileRowOffset     = (tileOffset % filterW);
            size_t tileColumnOffset  = (tileOffset / filterW);

            size_t inputTile       = row % (outputW * outputH);
            size_t inputTileRow    = (inputTile % outputW) * strideW;
            size_t inputTileColumn = (inputTile / outputW) * strideH;

            size_t inputRow    = (inputTileRow + tileRowOffset);
            size_t inputColumn = (inputTileColumn + tileColumnOffset);

            if(inputRow < padWidth || (inputRow >= (padWidth + w)))
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            inputRow -= padWidth;

            if(inputColumn < padHeight || (inputColumn >= (padHeight + h)))
            {
                resultView({row, column}) = 0.0;
                continue;
            }

            inputColumn -= padHeight;

            resultView(Dimension(row, column)) =
                inputView(Dimension(inputRow, inputColumn, inputFeatureMap, miniBatch));
        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> inputView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t w;
    size_t h;

public:
    size_t outputW;
    size_t outputH;

public:
    size_t filterW;
    size_t filterH;

public:
    size_t strideW;
    size_t strideH;

public:
    size_t padWidth;
    size_t padHeight;


};

template<typename PrecisionType>
Matrix gatherReverseConvolutionGradientsInput(const Matrix& input,
    const Matrix& deltas, const Dimension& filterSize, const Dimension& filterStride,
    const Dimension& padding, const PrecisionType& precisionType)
{
    assert(input.precision() == PrecisionType());

    size_t w                = input.size()[0];
    size_t h                = input.size()[1];
    size_t inputFeatureMaps = input.size()[2];
    size_t miniBatches      = input.size()[3];

    size_t filterW = filterSize[0];
    size_t filterH = filterSize[1];

    size_t strideW = filterStride[0];
    size_t strideH = filterStride[1];

    size_t padWidth  = padding[0];
    size_t padHeight = padding[1];

    size_t outputW = computeOutputSize(w, filterW, strideW, padWidth);
    size_t outputH = computeOutputSize(h, filterH, strideH, padHeight);

    size_t s = deltas.size()[0];
    size_t r = deltas.size()[1];

    assert(s == outputW);
    assert(r == outputH);

    size_t rows    = miniBatches * outputW * outputH;
    size_t columns = inputFeatureMaps * (filterW * filterH);

    Matrix result({rows, columns}, precisionType);

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    size_t elements = rows * columns;

    auto lambda = ReverseGradientsReshapeInput<NativeType>{resultView, inputView, elements,
        rows, w, h, outputW, outputH, filterW, filterH, strideW, strideH, padWidth, padHeight};

    parallel::multiBulkSynchronousParallel(lambda);

    return result;
}

template<typename NativeType>
class ReverseGradientsReshapeDeltas
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t k = row;

            size_t n = column / (outputW * outputH);

            size_t filterTile = column % (outputW * outputH);

            size_t w = filterTile % outputW;
            size_t h = filterTile / outputW;

            resultView(Dimension(row, column)) = deltasView(Dimension(w, h, k, n));

        }
    }

public:
    MatrixView<NativeType>      resultView;
    ConstMatrixView<NativeType> deltasView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t outputW;
    size_t outputH;

};

template<typename PrecisionType>
Matrix gatherReverseConvolutionGradientsDeltas(const Matrix& deltas, const PrecisionType& precisionType)
{
    assert(deltas.precision() == PrecisionType());

    size_t outputW           = deltas.size()[0];
    size_t outputH           = deltas.size()[1];
    size_t outputFeatureMaps = deltas.size()[2];
    size_t miniBatches       = deltas.size()[3];

    size_t rows    = outputFeatureMaps;
    size_t columns = outputW * outputH * miniBatches;

    Matrix result({rows, columns}, precisionType);

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> deltasView(deltas);

    size_t elements = rows * columns;

    auto lambda = ReverseGradientsReshapeDeltas<NativeType>{resultView,
        deltasView, elements, rows, outputW, outputH};

    parallel::multiBulkSynchronousParallel(lambda);

    return result;
}

template<typename NativeType>
class ReverseGradientsReshapeResult
{
public:
    CUDA_DECORATOR void operator()(parallel::ThreadGroup threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t w =  column % filterW;
            size_t h = (column / filterW) % filterH;

            size_t inputMap  = (column / (filterW * filterH));
            size_t outputMap = row;

            gradientsView(Dimension(filterW - w - 1, filterH - h - 1, inputMap, outputMap)) =
                reshapedGradientsView(Dimension(row, column));
        }
    }

public:
    MatrixView<NativeType>      gradientsView;
    ConstMatrixView<NativeType> reshapedGradientsView;

public:
    size_t elements;

public:
    size_t rows;

public:
    size_t filterW;
    size_t filterH;


};

template<typename PrecisionType>
void scatterReverseConvolutionGradientsResult(Matrix& gradients, const Matrix& reshapedGradients, const PrecisionType& precisionType)
{
    assert(gradients.precision()         == PrecisionType());
    assert(reshapedGradients.precision() == PrecisionType());

    size_t outputMaps = gradients.size()[3];
    size_t inputMaps  = gradients.size()[2];
    size_t filterH    = gradients.size()[1];
    size_t filterW    = gradients.size()[0];

    size_t rows = outputMaps;
    size_t columns = inputMaps * filterH * filterW;

    assert(rows    == reshapedGradients.size()[0]);
    assert(columns == reshapedGradients.size()[1]);

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      gradientsView(gradients);
    ConstMatrixView<NativeType> reshapedGradientsView(reshapedGradients);

    size_t elements = filterW * filterH * inputMaps * outputMaps;

    auto lambda = ReverseGradientsReshapeResult<NativeType>{gradientsView, reshapedGradientsView, elements, rows, filterW, filterH};

    parallel::multiBulkSynchronousParallel(lambda);
}

template<typename PrecisionType>
void genericReverseConvolutionGradientsOverPrecisions(Matrix& gradients, const Matrix& input,
    const Matrix& deltas, const Dimension& stride, const Dimension& padding, double alpha, const std::tuple<PrecisionType>& )
{
    assert(gradients.precision() == PrecisionType());
    assert(deltas.precision()    == PrecisionType());
    assert(input.precision()     == PrecisionType());

    auto inputFoldedTime  = foldTime(input);
    auto deltasFoldedTime = foldTime(deltas);

    // gather the inputs and deltas
    auto reshapedInput  = gatherReverseConvolutionGradientsInput (inputFoldedTime,  deltasFoldedTime, gradients.size(), stride, padding, PrecisionType());
    auto reshapedDeltas = gatherReverseConvolutionGradientsDeltas(deltasFoldedTime, PrecisionType());

    // then multiply like forward convolution
    auto reshapedGradients = gemm(Matrix(reshapedDeltas), false, alpha, reshapedInput, false);

    scatterReverseConvolutionGradientsResult(gradients, reshapedGradients, PrecisionType());
}

template<typename PossiblePrecisions>
void genericReverseConvolutionGradientsOverPrecisions(Matrix& gradients, const Matrix& input,
    const Matrix& deltas, const Dimension& stride, const Dimension& padding, double alpha, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(deltas.precision() == PossiblePrecisionType())
    {
        return genericReverseConvolutionGradientsOverPrecisions(gradients, input, deltas, stride, padding, alpha, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return genericReverseConvolutionGradientsOverPrecisions(gradients, input, deltas, stride, padding, alpha, RemainingPrecisions());
    }
}

void genericReverseConvolutionGradients(Matrix& gradients, const Matrix& input, const Matrix& deltas,
    const Dimension& stride, const Dimension& padding, double alpha)
{
    genericReverseConvolutionGradientsOverPrecisions(gradients, input, deltas, stride, padding, alpha, AllPrecisions());
}

}

void reverseConvolutionGradients(Matrix& gradients, const Matrix& inputs, const Matrix& deltas,
    const Dimension& stride, const Dimension& padding, double a)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            padding[1],       // zero-padding height
            padding[0],       // zero-padding width
            stride[1],        // vertical filter stride
            stride[0],        // horizontal filter stride
            1,                // upscale the input in x-direction
            1,                // upscale the input in y-direction
            CudnnLibrary::CUDNN_CONVOLUTION // convolution mode
        );

        CudnnFilterDescriptor gradientDescriptor(gradients);
        CudnnTensorDescriptor inputDescriptor(inputs);
        CudnnTensorDescriptor deltasDescriptor(deltas);

        CudnnScalar alpha(a,   deltas.precision());
        CudnnScalar beta( 0.0, deltas.precision());

        CudnnLibrary::cudnnConvolutionBackwardFilter(
            alpha.data(),
            inputDescriptor.descriptor(),
            inputDescriptor.data(),
            deltasDescriptor.descriptor(),
            deltasDescriptor.data(),
            convolutionDescriptor,
            beta.data(),
            gradientDescriptor.descriptor(),
            gradientDescriptor.data());

        CudnnLibrary::cudnnDestroyConvolutionDescriptor(convolutionDescriptor);
    }
    else
    {
        genericReverseConvolutionGradients(gradients, inputs, deltas, stride, padding, a);
    }
}

void reverseConvolutionGradients(Matrix& gradients, const Matrix& inputs, const Matrix& deltas,
    const Dimension& stride, const Dimension& padding)
{
    return reverseConvolutionGradients(gradients, inputs, deltas, stride, padding, 1.0);
}

}
}



