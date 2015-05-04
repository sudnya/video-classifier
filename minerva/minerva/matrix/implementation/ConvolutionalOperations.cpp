
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

Dimension getForwardConvolutionOutputSize(const Dimension& inputSize, const Dimension& filterSize, const Dimension& filterStride)
{
    size_t width   = computeOutputSize(inputSize[0], filterSize[0], filterStride[0], 0);
    size_t height  = computeOutputSize(inputSize[1], filterSize[1], filterStride[1], 0);
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
Matrix gatherForwardConvolutionInputOverPrecisions(const Matrix& input, const Matrix& filter, const Dimension& filterStride,
    const std::tuple<PrecisionType>& )
{
    assert(input.precision() == PrecisionType());
    assert(input.precision() == filter.precision());

    size_t p = computeOutputSize(input.size()[1], filter.size()[1], filterStride[1], 0);
    size_t q = computeOutputSize(input.size()[0], filter.size()[0], filterStride[0], 0);

    size_t r = filter.size()[1];
    size_t s = filter.size()[0];

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

            resultView({row, column}) = inputView({inputRow, inputColumn, featureMap, miniBatch});
        }
    });

    return result;
}

template<typename PossiblePrecisions>
Matrix gatherForwardConvolutionInputOverPrecisions(const Matrix& input, const Matrix& filter, const Dimension& stride,
    const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(input.precision() == PossiblePrecisionType())
    {
        return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, RemainingPrecisions());
    }

}

Matrix gatherForwardConvolutionInput(const Matrix& input, const Matrix& filter, const Dimension& stride)
{
    return gatherForwardConvolutionInputOverPrecisions(input, filter, stride, AllPrecisions());
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

void genericForwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter, const Dimension& stride)
{
    auto reshapedInput  = gatherForwardConvolutionInput(input, filter, stride);
    auto reshapedFilter = gatherForwardConvolutionFilter(filter);

    auto reshapedResult = gemm(reshapedFilter, reshapedInput);

    scatterForwardConvolutionResult(result, reshapedResult);
}

}

void forwardConvolution(Matrix& result, const Matrix& input, const Matrix& filter, const Dimension& stride)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            0,                // zero-padding height
            0,                // zero-padding width
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
        genericForwardConvolution(result, input, filter, stride);
    }
}

Matrix forwardConvolution(const Matrix& input, const Matrix& filter, const Dimension& stride)
{
    Matrix result(getForwardConvolutionOutputSize(input.size(), filter.size(), stride));

    forwardConvolution(result, input, filter, stride);

    return result;
}

namespace
{

template<typename PrecisionType>
Matrix gatherReverseConvolutionDeltasInput(const Matrix& deltas, const Matrix& filter, const Dimension& filterStride, const PrecisionType& )
{
    typedef typename PrecisionType::type NativeType;

    // zero fill the deltas to full convolution
    size_t padWidth  = filter.size()[0] - 1;
    size_t padHeight = filter.size()[1] - 1;

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

            if(inputRow < padWidth || (inputRow > (padWidth + w)))
            {
                resultView({row, columns}) = 0.0;
                continue;
            }

            inputRow -= padWidth;

            if(inputColumn < padHeight || (inputColumn > (padHeight + h)))
            {
                resultView({row, columns}) = 0.0;
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
    const Dimension& stride, const Matrix& deltas, const std::tuple<PrecisionType>& )
{
    assert(filter.precision()       == PrecisionType());
    assert(deltas.precision()       == PrecisionType());
    assert(resultDeltas.precision() == PrecisionType());

    // flip the filter
    auto reshapedInputDeltas = gatherReverseConvolutionDeltasInput(deltas, filter, stride, PrecisionType());
    auto reshapedFilter      = gatherReverseConvolutionDeltasFilter(filter, PrecisionType());

    auto reshapedResultDeltas = gemm(reshapedFilter, reshapedInputDeltas);

    // then multiply like forward convolution
    scatterReverseConvolutionDeltasResult(resultDeltas, reshapedResultDeltas, PrecisionType());
}

template<typename PossiblePrecisions>
void genericReverseConvolutionDeltasOverPrecisions(Matrix& resultDeltas, const Matrix& filter,
    const Dimension& stride, const Matrix& deltas, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(deltas.precision() == PossiblePrecisionType())
    {
        return genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride, deltas, std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        return genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride, deltas, RemainingPrecisions());
    }
}

void genericReverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter, const Dimension& stride, const Matrix& deltas)
{
    genericReverseConvolutionDeltasOverPrecisions(resultDeltas, filter, stride, deltas, AllPrecisions());
}

}

void reverseConvolutionDeltas(Matrix& resultDeltas, const Matrix& filter, const Dimension& stride, const Matrix& deltas)
{
    if(CudnnLibrary::loaded())
    {
        CudnnLibrary::cudnnConvolutionDescriptor_t convolutionDescriptor;

        CudnnLibrary::cudnnCreateConvolutionDescriptor(&convolutionDescriptor);

        CudnnLibrary::cudnnSetConvolution2dDescriptor(convolutionDescriptor,
            0,                // zero-padding height
            0,                // zero-padding width
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
        genericReverseConvolutionDeltas(resultDeltas, filter, stride, deltas);
    }
}

namespace
{

Dimension getReverseConvolutionDeltasSize(const Dimension& filterSize, const Dimension& deltaSize)
{
    Dimension result;

    assert(false  && "not implemented");

    return result;
}

}

Matrix reverseConvolutionDeltas(const Matrix& filter, const Dimension& stride, const Matrix& deltas)
{
    Matrix result(getReverseConvolutionDeltasSize(filter.size(), deltas.size()));

    reverseConvolutionDeltas(result, filter, stride, deltas);

    return result;
}

void reverseConvolutionGradients(Matrix& gradients, const Matrix& filter, const Matrix& inputs, const Matrix& deltas)
{
    // TODO
}

Matrix reverseConvolutionGradients(const Matrix& filter, const Matrix& inputs, const Matrix& deltas)
{
    Matrix result(filter.size());

    reverseConvolutionGradients(result, filter, inputs, deltas);

    return result;
}

}
}



