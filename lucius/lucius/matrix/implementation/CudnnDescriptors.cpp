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

#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <cassert>

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

void* CudnnFilterDescriptor::data()
{
    return _filter->data();
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Matrix& tensor)
: _tensor(std::make_unique<Matrix>(tensor))
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetTensor4dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        getCudnnDataType(_tensor->precision()), // image data type
        _tensor->size()[3],        // number of inputs (batch size)
        _tensor->size()[2],        // number of input feature maps
        _tensor->size()[1],        // height of input section
        _tensor->size()[0]         // width of input section
    );
}

CudnnTensorDescriptor::CudnnTensorDescriptor(const Dimension& size)
{
    CudnnLibrary::cudnnCreateTensorDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetTensor4dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_TENSOR_NCHW,
        CudnnLibrary::CUDNN_DATA_FLOAT, // image data type
        size[3],        // number of inputs (batch size)
        size[2],        // number of input feature maps
        size[1],        // height of input section
        size[0]         // width of input section
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

CudnnPooling2dDescriptor::CudnnPooling2dDescriptor(size_t inputW, size_t inputH, size_t padW,
    size_t padH, size_t poolW, size_t poolH)
{
    CudnnLibrary::cudnnCreatePoolingDescriptor(&_descriptor);

    CudnnLibrary::cudnnSetPooling2dDescriptor(_descriptor,
        CudnnLibrary::CUDNN_POOLING_MAX,
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

}

}



