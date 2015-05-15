
// Minerva Includes
#include <minerva/matrix/interface/ConvolutionalOperations.h>
#include <minerva/matrix/interface/CudnnLibrary.h>

#include <minerva/matrix/interface/BlasOperations.h>
#include <minerva/matrix/interface/MatrixView.h>
#include <minerva/matrix/interface/Matrix.h>

#include <minerva/parallel/interface/MultiBulkSynchronousParallel.h>

#include <minerva/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <vector>

namespace minerva
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

Dimension forwardConvolutionOutputSize(const Dimension& inputSize, const Dimension& filterSize, const Dimension& filterStride, const Dimension& padding)
{
    return {computeOutputSize(inputSize[0], filterSize[0], filterStride[0], padding[0]),
            computeOutputSize(inputSize[1], filterSize[1], filterStride[1], padding[1]),
            filterSize[3],
            inputSize[3]};
}

namespace
{

Dimension getForwardConvolutionOutputSize(const Dimension& inputSize, const Dimension& filterSize, const Dimension& filterStride, const Dimension& padding)
{
    size_t width   = computeOutputSize(inputSize[0], filterSize[0], filterStride[0], padding[0]);
    size_t height  = computeOutputSize(inputSize[1], filterSize[1], filterStride[1], padding[1]);
    size_t outputs = filterSize[3];

    return Dimension({width, height, outputs, inputSize[3]});
}

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
    const void* data() const
    {
        return _filter.data();
    }

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

    const void* data() const
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

        _data.resize(bytes);
    }

public:
    CudnnLibrary::cudnnConvolutionFwdAlgo_t algorithm()
    {
        return _algorithm;
    }

    void* data()
    {
        return _data.data();
    }

    size_t size() const
    {
        return _data.size();
    }

private:
    CudnnLibrary::cudnnConvolutionFwdAlgo_t _algorithm;

private:
    std::vector<uint8_t> _data;

};


template<typename PrecisionType>
Matrix gatherForwardConvolutionInputOverPrecisions(const Matrix& input, const Matrix& filter, const Dimension& filterStride, const Dimension& padding,
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

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
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

            resultView({row, column}) = inputView({inputRow, inputColumn, featureMap, miniBatch});
        }
    });

    return result;
}

template<typename PossiblePrecisions>
Matrix gatherForwardConvolutionInputOverPrecisions(const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(input.precision() == PossiblePrecisionType())
    {
        return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, padding, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, padding, RemainingPrecisions());
    }

}

Matrix gatherForwardConvolutionInput(const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding)
{
    return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, padding, AllPrecisions());
}

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

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
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

            resultView({row, column}) = filterView({filterR - r - 1, filterS - s - 1, c, k});
        }
    });

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

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t w = element % width;
            size_t h = (element / width) % height;
            size_t featureMap = (element / (width * height)) % featureMaps;
            size_t miniBatch  = (element / (width * height * featureMaps));

            size_t row    = featureMap;
            size_t column = w + h * width + miniBatch * width * height;

            resultView({w, h, featureMap, miniBatch}) = inputView({row, column});
        }
    });
}

template<typename PossiblePrecisions>
void scatterForwardConvolutionResultOverPrecisions(Matrix& result, const Matrix& input,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(input.precision() == PossiblePrecisionType())
    {
        return scatterForwardConvolutionResultOverPrecisions(result, input, std::tuple<PossiblePrecisionType>());
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

void genericForwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding)
{
    auto reshapedInput  = gatherForwardConvolutionInput(input, filter, stride, padding);
    auto reshapedFilter = gatherForwardConvolutionFilter(filter);

    auto reshapedResult = gemm(reshapedFilter, reshapedInput);

    scatterForwardConvolutionResult(result, reshapedResult);
}

}

void forwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            padding[0],                // zero-padding height
            padding[1],                // zero-padding width
            stride[0],        // vertical filter stride
            stride[1],        // horizontal filter stride
            1,                // upscale the input in x-direction
            1,                // upscale the input in y-direction
            CudnnLibrary::CUDNN_CONVOLUTION // convolution mode
        );

        CudnnFilterDescriptor filterDescriptor(filter);
        CudnnTensorDescriptor inputDescriptor(input);
        CudnnTensorDescriptor resultDescriptor(result);

        CudnnScalar alpha(1.0, input.precision());
        CudnnScalar beta( 1.0, input.precision());

        CudnnForwardWorkspace workspace(inputDescriptor, filterDescriptor, convolutionDescriptor, resultDescriptor);

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

Matrix forwardConvolution(const Matrix& input, const Matrix& filter, const Dimension& stride, const Dimension& padding)
{
    Matrix result(getForwardConvolutionOutputSize(input.size(), filter.size(), stride, padding), input.precision());

    forwardConvolution(result, input, filter, stride, padding);

    return result;
}

namespace
{

template<typename PrecisionType>
Matrix gatherReverseConvolutionDeltasInput(const Matrix& deltas, const Matrix& filter, const Dimension& filterStride,
    const Dimension& padding, const PrecisionType& )
{
    typedef typename PrecisionType::type NativeType;

    // zero fill the deltas to full convolution
    size_t padWidth  = padding[0];
    size_t padHeight = padding[1];

    size_t w = deltas.size()[0];
    size_t h = deltas.size()[1];

    size_t q = computeOutputSize(deltas.size()[0], filter.size()[0], filterStride[0], padWidth);
    size_t p = computeOutputSize(deltas.size()[1], filter.size()[1], filterStride[1], padHeight);

    size_t r = filter.size()[1];
    size_t s = filter.size()[0];

    size_t v = filterStride[0];
    size_t u = filterStride[1];

    size_t rows    = deltas.size()[2] * filter.size()[0] * filter.size()[1];
    size_t columns = deltas.size()[3] * p * q;

    Matrix result({rows, columns}, deltas.precision());

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> deltasView(deltas);

    size_t elements = rows * columns;

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
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

            resultView({row, column}) = deltasView({inputRow, inputColumn, featureMap, miniBatch});
        }
    });

    return result;
}

template<typename PrecisionType>
Matrix gatherReverseConvolutionDeltasFilter(const Matrix& filter, const PrecisionType& )
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

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
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

            resultView({row, column}) = filterView({r, s, c, k});
        }
    });

    return result;
}

template<typename PrecisionType>
void scatterReverseConvolutionDeltasResult(Matrix& result, const Matrix& input, const PrecisionType& )
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

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t w = element % width;
            size_t h = (element / width) % height;
            size_t featureMap = (element / (width * height)) % featureMaps;
            size_t miniBatch  = (element / (width * height * featureMaps));

            size_t row    = featureMap;
            size_t column = w + h * width + miniBatch * width * height;

            resultView({w, h, featureMap, miniBatch}) = inputView({row, column});
        }
    });

}

template<typename PrecisionType>
void genericReverseConvolutionDeltasOverPrecisions(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const Dimension& padding, const std::tuple<PrecisionType>& )
{
    assert(filter.precision()       == PrecisionType());
    assert(deltas.precision()       == PrecisionType());
    assert(resultDeltas.precision() == PrecisionType());

    // flip the filter
    auto reshapedInputDeltas = gatherReverseConvolutionDeltasInput(deltas, filter, stride, padding, PrecisionType());
    auto reshapedFilter      = gatherReverseConvolutionDeltasFilter(filter, PrecisionType());

    auto reshapedResultDeltas = gemm(reshapedFilter, reshapedInputDeltas);

    // then multiply like forward convolution
    scatterReverseConvolutionDeltasResult(resultDeltas, reshapedResultDeltas, PrecisionType());
}

template<typename PossiblePrecisions>
void genericReverseConvolutionDeltasOverPrecisions(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const Dimension& padding, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(deltas.precision() == PossiblePrecisionType())
    {
        return genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride, deltas, padding, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride, deltas, padding, RemainingPrecisions());
    }
}

void genericReverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter, const Dimension& stride, const Matrix& deltas, const Dimension& padding)
{
    genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride, deltas, padding, AllPrecisions());
}

}

void reverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter, const Dimension& stride, const Matrix& deltas, const Dimension& padding)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            padding[0],                // zero-padding height
            padding[1],                // zero-padding width
            stride[0],        // vertical filter stride
            stride[1],        // horizontal filter stride
            1,                // upscale the input in x-direction
            1,                // upscale the input in y-direction
            CudnnLibrary::CUDNN_CONVOLUTION // convolution mode
        );

        CudnnFilterDescriptor filterDescriptor(filter);
        CudnnTensorDescriptor deltasDescriptor(deltas);
        CudnnTensorDescriptor resultDescriptor(resultDeltas);

        CudnnScalar alpha(1.0, deltas.precision());
        CudnnScalar beta( 1.0, deltas.precision());

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

Dimension getReverseConvolutionDeltasSize(const Dimension& filterSize, const Dimension& filterStride, const Dimension& deltaSize, const Dimension& padding)
{
    size_t width   = computeOutputSize(deltaSize[0], filterSize[0], filterStride[0], padding[0]);
    size_t height  = computeOutputSize(deltaSize[1], filterSize[1], filterStride[1], padding[1]);
    size_t outputs = filterSize[3];

    return Dimension({width, height, outputs, deltaSize[3]});
}

}

Matrix reverseConvolutionDeltas(const Matrix& filter, const Dimension& stride, const Matrix& deltas, const Dimension& padding)
{
    Matrix result(getReverseConvolutionDeltasSize(filter.size(), stride, deltas.size(), padding), deltas.precision());

    reverseConvolutionDeltas(result, filter, stride, deltas, padding);

    return result;
}

namespace
{

template<typename PrecisionType>
Matrix gatherReverseConvolutionGradientsInput(const Matrix& input, const Matrix& deltas, const Dimension& filterStride,
    const Dimension& padding, const PrecisionType& precisionType)
{
    assert(input.precision() == PrecisionType());

    size_t w           = input.size()[0];
    size_t h           = input.size()[1];
    size_t featureMaps = input.size()[2];
    size_t miniBatches = input.size()[3];

    size_t v = filterStride[0];
    size_t u = filterStride[1];

    size_t padWidth  = padding[0];
    size_t padHeight = padding[1];

    size_t q = computeOutputSize(input.size()[0], deltas.size()[0], filterStride[0], padding[0]);
    size_t p = computeOutputSize(input.size()[1], deltas.size()[1], filterStride[1], padding[1]);

    size_t s = deltas.size()[0];
    size_t r = deltas.size()[1];

    size_t rows    = miniBatches * (s * r);
    size_t columns = featureMaps * (p * q);

    Matrix result({rows, columns}, precisionType);

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> inputView(input);

    size_t elements = rows * columns;

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t miniBatch   = (row    / (s * r));
            size_t featureMap  = (column / (p * q));

            size_t tileOffset  = row % (s * r);
            size_t tileRow     = (tileOffset % s) * v;
            size_t tileColumn  = (tileOffset / s) * u;

            size_t inputTile   = column % (p * q);

            size_t inputRow    = ((inputTile % q) + tileRow);
            size_t inputColumn = ((inputTile / q) + tileColumn);

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

            resultView({row, column}) = inputView({inputRow, inputColumn, featureMap, miniBatch});
        }
    });

    return result;
}

template<typename PrecisionType>
Matrix gatherReverseConvolutionGradientsDeltas(const Matrix& deltas, const PrecisionType& precisionType)
{
    assert(deltas.precision() == PrecisionType());

    size_t rows    = deltas.size()[2];
    size_t columns = deltas.size()[3] * deltas.size()[0] * deltas.size()[1];

    Matrix result({rows, columns}, precisionType);

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      resultView(result);
    ConstMatrixView<NativeType> deltasView(deltas);

    size_t elements = rows * columns;

    size_t deltasW = deltas.size()[0];
    size_t deltasH = deltas.size()[1];

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t row    = element % rows;
            size_t column = element / rows;

            size_t k = row;

            size_t n = column / (deltasW * deltasH);

            size_t filterTile = column % (deltasW * deltasH);

            size_t w = filterTile % deltasW;
            size_t h = filterTile / deltasW;

            resultView({row, column}) = deltasView({w, h, k, n});

        }
    });

    return result;
}

template<typename PrecisionType>
void scatterReverseConvolutionGradientsResult(Matrix& gradients, const Matrix& reshapedGradients, const PrecisionType& precisionType)
{
    assert(gradients.precision()         == PrecisionType());
    assert(reshapedGradients.precision() == PrecisionType());

    size_t outputMaps = gradients.size()[3];
    size_t inputMaps  = gradients.size()[2];
    size_t height     = gradients.size()[1];
    size_t width      = gradients.size()[0];

    size_t rows = reshapedGradients.size()[0];

    typedef typename PrecisionType::type NativeType;

    MatrixView<NativeType>      gradientsView(gradients);
    ConstMatrixView<NativeType> reshapedGradientsView(reshapedGradients);

    size_t elements = width * height * inputMaps * outputMaps;

    parallel::multiBulkSynchronousParallel([=](parallel::ThreadGroup threadGroup)
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            size_t w =  element % width;
            size_t h = (element / width) % height;

            size_t inputMap  = (element / (width * height)) % inputMaps;
            size_t outputMap = (element / (width * height * inputMaps));

            size_t row    = element % rows;
            size_t column = element / rows;

            gradientsView({width - w - 1, height - h - 1, inputMap, outputMap}) = reshapedGradientsView({row, column});
        }
    });
}

template<typename PrecisionType>
void genericReverseConvolutionGradientsOverPrecisions(Matrix& gradients, const Matrix& input,
    const Matrix& deltas, const Dimension& stride, const Dimension& padding, const std::tuple<PrecisionType>& )
{
    assert(gradients.precision() == PrecisionType());
    assert(deltas.precision()    == PrecisionType());
    assert(input.precision()     == PrecisionType());

    // gather the inputs and deltas
    auto reshapedInput  = gatherReverseConvolutionGradientsInput (input,  deltas, stride, padding, PrecisionType());
    auto reshapedDeltas = gatherReverseConvolutionGradientsDeltas(deltas, PrecisionType());

    // then multiply like forward convolution
    auto reshapedGradients = gemm(reshapedDeltas, reshapedInput);

    scatterReverseConvolutionGradientsResult(gradients, reshapedGradients, PrecisionType());
}

template<typename PossiblePrecisions>
void genericReverseConvolutionGradientsOverPrecisions(Matrix& gradients, const Matrix& input,
    const Matrix& deltas, const Dimension& stride, const Dimension& padding, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(deltas.precision() == PossiblePrecisionType())
    {
        return genericReverseConvolutionGradientsOverPrecisions(gradients, input, deltas, stride, padding, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return genericReverseConvolutionGradientsOverPrecisions(gradients, input, deltas, stride, padding, RemainingPrecisions());
    }
}

void genericReverseConvolutionGradients(Matrix& gradients, const Matrix& input, const Matrix& deltas, const Dimension& stride, const Dimension& padding)
{
    genericReverseConvolutionGradientsOverPrecisions(gradients, input, deltas, stride, padding, AllPrecisions());
}

}

void reverseConvolutionGradients(Matrix& gradients, const Matrix& inputs, const Matrix& deltas, const Dimension& stride, const Dimension& padding)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            padding[0],       // zero-padding height
            padding[1],       // zero-padding width
            stride[0],        // vertical filter stride
            stride[1],        // horizontal filter stride
            1,                // upscale the input in x-direction
            1,                // upscale the input in y-direction
            CudnnLibrary::CUDNN_CONVOLUTION // convolution mode
        );

        CudnnFilterDescriptor gradientDescriptor(gradients);
        CudnnTensorDescriptor inputDescriptor(inputs);
        CudnnTensorDescriptor deltasDescriptor(deltas);

        CudnnScalar alpha(1.0, deltas.precision());
        CudnnScalar beta( 1.0, deltas.precision());

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
        genericReverseConvolutionGradients(gradients, inputs, deltas, stride, padding);
    }
}

namespace
{

Dimension getReverseConvolutionGradientsSize(const Dimension& inputSize, const Dimension& deltaSize, const Dimension& filterStride, const Dimension& padding)
{
    size_t width   = computeOutputSize(inputSize[0], deltaSize[0], filterStride[0], padding[0]);
    size_t height  = computeOutputSize(inputSize[1], deltaSize[1], filterStride[1], padding[1]);

    return Dimension({width, height, inputSize[2], deltaSize[2]});
}

}

Matrix reverseConvolutionGradients(const Matrix& inputs, const Matrix& deltas, const Dimension& stride, const Dimension& padding)
{
    Matrix result(getReverseConvolutionGradientsSize(inputs.size(), deltas.size(), stride, padding), inputs.precision());

    reverseConvolutionGradients(result, inputs, deltas, stride, padding);

    return result;
}

}
}



