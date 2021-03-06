/*  \file   CudnnDescriptors.h
    \date   Thursday August 15, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the cudnn descriptor C++ wrappers.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <vector>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;             } }
namespace lucius { namespace matrix { class Dimension;          } }
namespace lucius { namespace matrix { class Precision;          } }
namespace lucius { namespace matrix { class Allocation;         } }
namespace lucius { namespace matrix { class RecurrentOpsHandle; } }

typedef struct cudnnTensorStruct*      cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct*      cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*     cudnnPoolingDescriptor_t;
typedef struct cudnnRNNStruct*         cudnnRNNDescriptor_t;
typedef struct cudnnDropoutStruct*     cudnnDropoutDescriptor_t;

namespace lucius
{

namespace matrix
{

class CudnnFilterDescriptor
{
public:
    CudnnFilterDescriptor(CudnnFilterDescriptor&&);
    CudnnFilterDescriptor(const Matrix& filter);

    ~CudnnFilterDescriptor();

public:
    cudnnFilterDescriptor_t descriptor() const;

public:
    Dimension getDimensions() const;

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
    CudnnTensorDescriptor(CudnnTensorDescriptor&&);
    CudnnTensorDescriptor(const Matrix& tensor);
    CudnnTensorDescriptor(const Dimension& size, const Dimension& stride,
        const Precision& precision);
    ~CudnnTensorDescriptor();

public:
    cudnnTensorDescriptor_t descriptor() const;
    cudnnTensorDescriptor_t& descriptor();
    void* data();
    size_t bytes() const;

public:
    Dimension getDimensions() const;

private:
    cudnnTensorDescriptor_t _descriptor;

private:
    std::unique_ptr<Matrix> _tensor;

};

class CudnnTensorDescriptorArray
{
public:
    CudnnTensorDescriptorArray(CudnnTensorDescriptorArray&&) = default;
    CudnnTensorDescriptorArray(void* data, const Dimension& size, const Dimension& strides,
        size_t timesteps, const Precision& precision);
    CudnnTensorDescriptorArray(const Dimension& size, const Dimension& strides,
        size_t timesteps, const Precision& precision);
    ~CudnnTensorDescriptorArray();

public:
    cudnnTensorDescriptor_t* descriptors();

public:
    void* data() const;

public:
    Dimension getDimensions() const;

public:
    std::string toString() const;

public:
    CudnnTensorDescriptorArray& operator=(const CudnnTensorDescriptorArray&) = delete;
    CudnnTensorDescriptorArray(const CudnnTensorDescriptorArray&) = delete;

private:
    std::vector<cudnnTensorDescriptor_t> _descriptors;

private:
    void* _data;

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

class CudnnBackwardDataWorkspace
{
public:
    CudnnBackwardDataWorkspace(const CudnnFilterDescriptor& filter,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnTensorDescriptor& inputDeltas);

public:
    int algorithm() const;
    void* data();
    size_t size() const;

private:
    int _algorithm;

private:
    std::unique_ptr<Allocation> _data;
};

class CudnnBackwardFilterWorkspace
{
public:
    CudnnBackwardFilterWorkspace(const CudnnTensorDescriptor& source,
        const CudnnTensorDescriptor& outputDeltas,
        cudnnConvolutionDescriptor_t convolutionDescriptor,
        const CudnnFilterDescriptor& filterGradient
        );

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

class CudnnRNNDescriptor
{
public:
    CudnnRNNDescriptor(const RecurrentOpsHandle&, const Precision& precision);
    ~CudnnRNNDescriptor();

public:
    cudnnRNNDescriptor_t descriptor() const;

public:
    cudnnRNNDescriptor_t _descriptor;
    cudnnDropoutDescriptor_t _dropoutDescriptor;
};

}

}


