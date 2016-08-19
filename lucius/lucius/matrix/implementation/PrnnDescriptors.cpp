/*  \file   PrnnDescriptors.cpp
    \date   Thursday August 15, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the prnn descriptor C++ wrappers.
*/

// Lucius Includes
#include <lucius/matrix/interface/PrnnDescriptors.h>
#include <lucius/matrix/interface/PrnnLibrary.h>
#include <lucius/matrix/interface/RecurrentOperations.h>

#include <lucius/matrix/interface/Precision.h>
#include <lucius/matrix/interface/Matrix.h>

#include <lucius/util/interface/memory.h>
#include <lucius/util/interface/debug.h>

namespace lucius
{

namespace matrix
{

PrnnTensorDescriptor::PrnnTensorDescriptor(PrnnTensorDescriptor&& tensor)
: _descriptor(tensor._descriptor), _tensor(std::move(tensor._tensor))
{
    tensor._descriptor = nullptr;
}

PrnnTensorDescriptor::PrnnTensorDescriptor(const Matrix& tensor)
: PrnnTensorDescriptor(tensor.size(), tensor.stride(), tensor.precision())
{
    _tensor = std::make_unique<Matrix>(tensor);
}

PrnnLibrary::prnnDataType_t getDatatype(const Precision& precision)
{
    if(precision == SinglePrecision())
    {
        return PrnnLibrary::PRNN_DATA_FLOAT;
    }
    else if(precision == DoublePrecision())
    {
        return PrnnLibrary::PRNN_DATA_DOUBLE;
    }
    else if(precision == HalfPrecision())
    {
        return PrnnLibrary::PRNN_DATA_HALF;
    }

    assertM(false, "Invalid precision.");
}

PrnnTensorDescriptor::PrnnTensorDescriptor(const Dimension& size, const Dimension& inputStrides,
    const Precision& precision)
{
    PrnnLibrary::prnnCreateTensorDescriptor(&descriptor());

    std::vector<int> dims(size.begin(), size.end());
    std::vector<int> strides(inputStrides.begin(), inputStrides.end());

    PrnnLibrary::prnnSetTensorNdDescriptor(descriptor(),
                                           getDatatype(precision),
                                           size.size(),
                                           dims.data(),
                                           strides.data());
}

PrnnTensorDescriptor::~PrnnTensorDescriptor()
{
    PrnnLibrary::prnnDestroyTensorDescriptor(descriptor());
}

prnnTensorDescriptor_t PrnnTensorDescriptor::descriptor() const
{
    return _descriptor;
}

prnnTensorDescriptor_t& PrnnTensorDescriptor::descriptor()
{
    return _descriptor;
}

void* PrnnTensorDescriptor::data()
{
    if(!_tensor)
    {
        return nullptr;
    }

    return _tensor->data();
}

static std::tuple<Dimension, Dimension> getSizeAndStride(prnnTensorDescriptor_t descriptor)
{
    int dims = 8;

    std::vector<int> size(dims);
    std::vector<int> stride(dims);

    PrnnLibrary::prnnDataType_t datatype;

    PrnnLibrary::prnnGetTensorNdDescriptor(descriptor,
                                           dims,
                                           &datatype,
                                           &dims,
                                           size.data(),
                                           stride.data());

    size.resize(dims);
    stride.resize(dims);

    Dimension sizeDim;
    Dimension strideDim;

    for(int i = 0; i < dims; ++i)
    {
        sizeDim.push_back(size[i]);
        strideDim.push_back(stride[i]);
    }

    return std::make_tuple(sizeDim, strideDim);
}

size_t PrnnTensorDescriptor::bytes() const
{
    if(descriptor() == nullptr)
    {
        return 0;
    }

    auto sizeAndStride = getSizeAndStride(descriptor());

    return std::get<0>(sizeAndStride).product();
}

Dimension PrnnTensorDescriptor::dimensions() const
{
    return std::get<0>(getSizeAndStride(descriptor()));
}

static PrnnLibrary::prnnRNNInputMode_t getInputMode(const RecurrentOpsHandle& handle)
{
    switch(handle.inputMode)
    {
        case RECURRENT_LINEAR_INPUT:
        {
            return PrnnLibrary::PRNN_LINEAR_INPUT;
        }
        case RECURRENT_SKIP_INPUT:
        {
            return PrnnLibrary::PRNN_SKIP_INPUT;
        }
        default: break;
    }

    assertM(false, "Invalid input mode.");
}

static PrnnLibrary::prnnDirectionMode_t getDirection(const RecurrentOpsHandle& handle)
{
    switch(handle.direction)
    {
    case RECURRENT_FORWARD:
    {
        return PrnnLibrary::PRNN_UNIDIRECTIONAL;
    }
    case RECURRENT_REVERSE:
    {
        return PrnnLibrary::PRNN_REVERSE;
    }
    case RECURRENT_BIDIRECTIONAL:
    {
        return PrnnLibrary::PRNN_BIDIRECTIONAL;
    }
    default: break;
    }

    assertM(false, "Invalid direction.");
}

static PrnnLibrary::prnnRNNMode_t getLayerType(const RecurrentOpsHandle& handle)
{
    switch(handle.layerType)
    {
    case RECURRENT_SIMPLE_TYPE:
    {
        return PrnnLibrary::PRNN_RNN_RELU;
    }
    case RECURRENT_GRU_TYPE:
    {
        return PrnnLibrary::PRNN_GRU;
    }
    case RECURRENT_LSTM_TYPE:
    {
        return PrnnLibrary::PRNN_LSTM;
    }
    default: break;
    }

    assertM(false, "Invalid layer type.");
}

PrnnRNNDescriptor::PrnnRNNDescriptor(const RecurrentOpsHandle& handle, const Precision& precision)
{
    PrnnLibrary::prnnCreateRNNDescriptor(&_descriptor);
    PrnnLibrary::prnnSetRNNDescriptor(_descriptor,
                                      handle.layerSize,
                                      handle.layers,
                                      nullptr,
                                      getInputMode(handle),
                                      getDirection(handle),
                                      getLayerType(handle),
                                      getDatatype(precision),
                                      PrnnLibrary::PRNN_BEST_BACKEND);
}

PrnnRNNDescriptor::~PrnnRNNDescriptor()
{
    PrnnLibrary::prnnDestroyRNNDescriptor(descriptor());
}

prnnRNNDescriptor_t PrnnRNNDescriptor::descriptor() const
{
    return _descriptor;
}

}

}


