/*  \file   CudnnDescriptors.h
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the cudnn descriptor C++ wrappers.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;     } }
namespace lucius { namespace matrix { class Dimension;  } }
namespace lucius { namespace matrix { class Precision;  } }
namespace lucius { namespace matrix { class Allocation; } }

typedef struct cudnnTensorStruct*      cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct*      cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*     cudnnPoolingDescriptor_t;

namespace lucius
{

namespace matrix
{

class CudnnFilterDescriptor
{
public:
    CudnnFilterDescriptor(const Matrix& filter);

    ~CudnnFilterDescriptor();

public:
    cudnnFilterDescriptor_t descriptor() const;

public:
    void* data();

private:
    cudnnFilterDescriptor_t _descriptor;

private:
    std::unique_ptr<Matrix> _filter;

};

class CudnnTensorDescriptor
{
public:
    CudnnTensorDescriptor(const Matrix& tensor);
    CudnnTensorDescriptor(const Dimension& size);
    ~CudnnTensorDescriptor();

public:
    cudnnTensorDescriptor_t descriptor() const;
    void* data();
    size_t bytes() const;

private:
    cudnnTensorDescriptor_t _descriptor;

private:
    std::unique_ptr<Matrix> _tensor;

};

class CudnnScalar
{
public:
    CudnnScalar(double value, const Precision& p);
    ~CudnnScalar();

public:
    void* data();

private:
    double _doubleValue;
    float  _floatValue;

private:
    std::unique_ptr<Precision> _precision;

};

class CudnnForwardWorkspace
{
public:
    CudnnForwardWorkspace(const CudnnTensorDescriptor& source, const CudnnFilterDescriptor& filter,
        cudnnConvolutionDescriptor_t convolutionDescriptor, const CudnnTensorDescriptor& result);

public:
    int algorithm() const;
    void* data();
    size_t size() const;

private:
    int _algorithm;

private:
    std::unique_ptr<Allocation> _data;
};

class CudnnPooling2dDescriptor
{
public:
    CudnnPooling2dDescriptor(size_t inputW, size_t inputH, size_t padW, size_t padH,
        size_t poolW, size_t poolH);
    ~CudnnPooling2dDescriptor();

public:
    cudnnPoolingDescriptor_t descriptor() const;

public:
    CudnnPooling2dDescriptor& operator=(const CudnnPooling2dDescriptor&) = delete;
    CudnnPooling2dDescriptor(const CudnnPooling2dDescriptor&) = delete;

private:
    cudnnPoolingDescriptor_t _descriptor;

};

}

}


