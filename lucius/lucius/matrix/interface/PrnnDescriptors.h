/*  \file   PrnnDescriptors.h
    \date   Thursday August 15, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the prnn descriptor C++ wrappers.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix;             } }
namespace lucius { namespace matrix { class Dimension;          } }
namespace lucius { namespace matrix { class Precision;          } }
namespace lucius { namespace matrix { class Allocation;         } }
namespace lucius { namespace matrix { class RecurrentOpsHandle; } }

typedef struct prnnTensorStruct*      prnnTensorDescriptor_t;
typedef struct prnnTensorStruct*      prnnFilterDescriptor_t;
typedef struct prnnConvolutionStruct* prnnConvolutionDescriptor_t;
typedef struct prnnPoolingStruct*     prnnPoolingDescriptor_t;
typedef struct prnnRNNStruct*         prnnRNNDescriptor_t;

namespace lucius
{

namespace matrix
{

class PrnnTensorDescriptor
{
public:
    PrnnTensorDescriptor(PrnnTensorDescriptor&&);
    PrnnTensorDescriptor(const Matrix& tensor);
    PrnnTensorDescriptor(const Dimension& size, const Dimension& strides,
        const Precision& precision);
    ~PrnnTensorDescriptor();

public:
    prnnTensorDescriptor_t descriptor() const;
    prnnTensorDescriptor_t& descriptor();

    void* data();
    size_t bytes() const;

    Dimension dimensions() const;

private:
    prnnTensorDescriptor_t _descriptor;

private:
    std::unique_ptr<Matrix> _tensor;

};

class PrnnRNNDescriptor
{
public:
    PrnnRNNDescriptor(const RecurrentOpsHandle&, const Precision& precision);
    ~PrnnRNNDescriptor();

public:
    PrnnRNNDescriptor(const PrnnRNNDescriptor& descriptor) = delete;
    PrnnRNNDescriptor& operator=(const PrnnRNNDescriptor& descriptor) = delete;

public:
    prnnRNNDescriptor_t descriptor() const;

public:
    prnnRNNDescriptor_t _descriptor;
};

}

}


