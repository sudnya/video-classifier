/*  \file   CudnnDescriptors.cpp
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the cudnn descriptor C++ wrappers.
*/

// Lucius Includes
#include <lucius/matrix/interface/CudnnDescriptors.h>
#include <lucius/matrix/interface/CudnnLibrary.h>
#include <lucius/matrix/interface/Precision.h>
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/RecurrentOperations.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <cassert>
#include <vector>

namespace lucius
{

namespace matrix
{

static CudnnLibrary::cudnnDataType_t getCudnnDataType(const Precision& precision)
{
    if(precision == DoublePrecision())
    {
        return CudnnLibrary::CUDNN_DATA_DOUBLE;
    }

    assert(precision == SinglePrecision());

    return CudnnLibrary::CUDNN_DATA_FLOAT;
}

CudnnFilterDescriptor::CudnnFilterDescriptor(const Matrix& filter)
: _filter(std::make_unique<Matrix>(filter))
{
    CudnnLibrary::cudnnCreateFilterDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetFilter4dDescriptor(_descriptor,
        getCudnnDataType(filter.precision()),
        filter.size()[3],
        filter.size()[2],
        filter.size()[1],
        filter.size()[0]);
}

CudnnFilterDescriptor::~CudnnFilterDescriptor()
{
    CudnnLibrary::cudnnDestroyFilterDescriptor(_descriptor);
}

cudnnFilterDescriptor_t CudnnFilterDescriptor::descriptor() const
{
    return _descriptor;
}

Dimension CudnnFilterDescriptor::dimensions() const
{
    return _filter->size();
}

void* CudnnFilterDescriptor::data()
{
    return _filter->data();
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Matrix& tensor)
: CudnnTensorDescriptor(tensor.size(), tensor.stride(), tensor.precision())
{
    _tensor = std::make_unique<Matrix>(tensor);
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Dimension& size, const Dimension& stride,
    const Precision& precision)
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    std::vector<int> sizeArray(size.begin(), size.end());
    std::vector<int> strideArray(stride.begin(), stride.end());

    CudnnLibrary::cudnnSetTensorNdDescriptor(_descriptor,
        getCudnnDataType(precision),
        size.size(),
        sizeArray.data(),
        strideArray.data()
    );
}

CudnnTensorDescriptor::~CudnnTensorDescriptor()
{
    CudnnLibrary::cudnnDestroyTensorDescriptor(_descriptor);
}

cudnnTensorDescriptor_t CudnnTensorDescriptor::descriptor() const
{
    return _descriptor;
}

cudnnTensorDescriptor_t& CudnnTensorDescriptor::descriptor()
{
    return _descriptor;
}

void* CudnnTensorDescriptor::data()
{
    return _tensor->data();
}

size_t CudnnTensorDescriptor::bytes() const
{
    return _tensor->elements() * _tensor->precision().size();
}

CudnnScalar::CudnnScalar(double value, const Precision& p)
: _doubleValue(value), _floatValue(value), _precision(std::make_unique<Precision>(p))
{

}

CudnnScalar::~CudnnScalar()
{

}

void* CudnnScalar::data()
{
    if(*_precision == SinglePrecision())
    {
        return &_floatValue;
    }
    else
    {
        return &_doubleValue;
    }
}

CudnnForwardWorkspace::CudnnForwardWorkspace(const CudnnTensorDescriptor& source,
    const CudnnFilterDescriptor& filter,
    cudnnConvolutionDescriptor_t convolutionDescriptor, const CudnnTensorDescriptor& result)
{
    CudnnLibrary::cudnnConvolutionFwdAlgo_t algorithm;

    CudnnLibrary::cudnnGetConvolutionForwardAlgorithm(
        source.descriptor(),
        filter.descriptor(),
        convolutionDescriptor,
        result.descriptor(),
        CudnnLibrary::CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, // TODO: make this a knob
        source.bytes(),
        &algorithm);

    _algorithm = algorithm;

    size_t bytes = 0;

    CudnnLibrary::cudnnGetConvolutionForwardWorkspaceSize(
        source.descriptor(),
        filter.descriptor(),
        convolutionDescriptor,
        result.descriptor(),
        algorithm,
        &bytes);

    _data = std::make_unique<Allocation>(bytes);
}

int CudnnForwardWorkspace::algorithm() const
{
    return _algorithm;
}

void* CudnnForwardWorkspace::data()
{
    return _data->data();
}

size_t CudnnForwardWorkspace::size() const
{
    return _data->size();
}

CudnnBackwardDataWorkspace::CudnnBackwardDataWorkspace(const CudnnFilterDescriptor& filter,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnTensorDescriptor& inputDeltas)
{
    CudnnLibrary::cudnnConvolutionBwdDataAlgo_t algorithm;

    CudnnLibrary::cudnnGetConvolutionBackwardDataAlgorithm(
        filter.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        inputDeltas.descriptor(),
        CudnnLibrary::CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, // TODO: make this a knob
        inputDeltas.bytes(),
        &algorithm);

    _algorithm = algorithm;

    size_t bytes = 0;

    CudnnLibrary::cudnnGetConvolutionBackwardDataWorkspaceSize(
        filter.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        inputDeltas.descriptor(),
        algorithm,
        &bytes);

    _data = std::make_unique<Allocation>(bytes);
}

int CudnnBackwardDataWorkspace::algorithm() const
{
    return _algorithm;
}

void* CudnnBackwardDataWorkspace::data()
{
    return _data->data();
}

size_t CudnnBackwardDataWorkspace::size() const
{
    return _data->size();
}

CudnnBackwardFilterWorkspace::CudnnBackwardFilterWorkspace(const CudnnTensorDescriptor& input,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnFilterDescriptor& filterGradient)
{
    CudnnLibrary::cudnnConvolutionBwdFilterAlgo_t algorithm;

    CudnnLibrary::cudnnGetConvolutionBackwardFilterAlgorithm(
        input.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        filterGradient.descriptor(),
        CudnnLibrary::CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, // TODO: make this a knob
        input.bytes(),
        &algorithm);

    _algorithm = algorithm;

    size_t bytes = 0;

    CudnnLibrary::cudnnGetConvolutionBackwardFilterWorkspaceSize(
        input.descriptor(),
        outputDeltas.descriptor(),
        convolutionDescriptor,
        filterGradient.descriptor(),
        algorithm,
        &bytes);

    _data = std::make_unique<Allocation>(bytes);
}

int CudnnBackwardFilterWorkspace::algorithm() const
{
    return _algorithm;
}

void* CudnnBackwardFilterWorkspace::data()
{
    return _data->data();
}

size_t CudnnBackwardFilterWorkspace::size() const
{
    return _data->size();
}

CudnnPooling2dDescriptor::CudnnPooling2dDescriptor(size_t inputW, size_t inputH, size_t padW,
    size_t padH, size_t poolW, size_t poolH)
{
    CudnnLibrary::cudnnCreatePoolingDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetPooling2dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_POOLING_MAX,
        CudnnLibrary::CUDNN_NOT_PROPAGATE_NAN,
        inputH,
        inputW,
        padH,
        padW,
        poolH,
        poolW);
}

CudnnPooling2dDescriptor::~CudnnPooling2dDescriptor()
{
    CudnnLibrary::cudnnDestroyPoolingDescriptor(_descriptor);
}

cudnnPoolingDescriptor_t CudnnPooling2dDescriptor::descriptor() const
{
    return _descriptor;
}

static CudnnLibrary::cudnnRNNInputMode_t getInputMode(const RecurrentOpsHandle& handle)
{
    switch(handle.inputMode)
    {
        case RECURRENT_LINEAR_INPUT:
        {
            return CudnnLibrary::CUDNN_LINEAR_INPUT;
        }
        case RECURRENT_SKIP_INPUT:
        {
            return CudnnLibrary::CUDNN_SKIP_INPUT;
        }
        default: break;
    }

    assertM(false, "Invalid input mode.");
}

static CudnnLibrary::cudnnDirectionMode_t getDirection(const RecurrentOpsHandle& handle)
{
    switch(handle.direction)
    {
    case RECURRENT_FORWARD:
    {
        return CudnnLibrary::CUDNN_UNIDIRECTIONAL;
    }
    case RECURRENT_REVERSE:
    {
        return CudnnLibrary::CUDNN_REVERSE;
    }
    case RECURRENT_BIDIRECTIONAL:
    {
        return CudnnLibrary::CUDNN_BIDIRECTIONAL;
    }
    default: break;
    }

    assertM(false, "Invalid direction.");
}

static CudnnLibrary::cudnnRNNMode_t getLayerType(const RecurrentOpsHandle& handle)
{
    switch(handle.layerType)
    {
    case RECURRENT_SIMPLE_TYPE:
    {
        return CudnnLibrary::CUDNN_RNN_RELU;
    }
    case RECURRENT_SIMPLE_TANH_TYPE:
    {
        return CudnnLibrary::CUDNN_RNN_TANH;
    }
    case RECURRENT_GRU_TYPE:
    {
        return CudnnLibrary::CUDNN_GRU;
    }
    case RECURRENT_LSTM_TYPE:
    {
        return CudnnLibrary::CUDNN_LSTM;
    }
    default: break;
    }

    assertM(false, "Invalid layer type.");
}

CudnnRNNDescriptor::CudnnRNNDescriptor(const RecurrentOpsHandle& handle,
    const Precision& precision)
{
    CudnnLibrary::cudnnCreateRNNDescriptor(&_descriptor);
    CudnnLibrary::cudnnSetRNNDescriptor(_descriptor,
                                      handle.layerSize,
                                      handle.layers,
                                      nullptr,
                                      getInputMode(handle),
                                      getDirection(handle),
                                      getLayerType(handle),
                                      getCudnnDataType(precision));
}

CudnnRNNDescriptor::~CudnnRNNDescriptor()
{
    CudnnLibrary::cudnnDestroyRNNDescriptor(descriptor());
}

cudnnRNNDescriptor_t CudnnRNNDescriptor::descriptor() const
{
    return _descriptor;
}

}

}



