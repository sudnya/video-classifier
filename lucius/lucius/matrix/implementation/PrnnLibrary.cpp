/*  \file   PrnnLibrary.cpp
    \date   July 31, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the PrnnLibrary class.
*/

// Lucius Includes
#include <lucius/matrix/interface/PrnnLibrary.h>

#include <lucius/parallel/interface/cuda.h>

#include <lucius/util/interface/Casts.h>

// Standard Library Includes
#include <stdexcept>

// System-Specific Includes
#include <dlfcn.h>

namespace lucius
{

namespace matrix
{

void PrnnLibrary::load()
{
    _interface.load();
}

bool PrnnLibrary::loaded()
{
    load();

    return _interface.loaded();
}

void PrnnLibrary::prnnCreateTensorDescriptor(prnnTensorDescriptor_t* tensorDesc)
{
    _check();

    auto status = (*_interface.prnnCreateTensorDescriptor)(tensorDesc);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnnCreateTensorDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnSetTensorNdDescriptor(prnnTensorDescriptor_t tensorDesc,
                                       prnnDataType_t         dataType,
                                       int                    nbDims,
                                       const int*             dimA,
                                       const int*             strideA)
{
    _check();

    auto status = (*_interface.prnnSetTensorNdDescriptor)(tensorDesc,
                                                          dataType,
                                                          nbDims,
                                                          dimA,
                                                          strideA);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnnSetTensorNdDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void PrnnLibrary::prnnGetTensorNdDescriptor(const prnnTensorDescriptor_t tensorDesc,
                                      int                          nbDimsRequested,
                                      prnnDataType_t*              dataType,
                                      int*                         nbDims,
                                      int*                         dimA,
                                      int*                         strideA)
{
    _check();

    auto status = (*_interface.prnnGetTensorNdDescriptor)(tensorDesc,
                                                          nbDimsRequested,
                                                          dataType,
                                                          nbDims,
                                                          dimA,
                                                          strideA);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnnGetTensorNdDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnDestroyTensorDescriptor(prnnTensorDescriptor_t tensorDesc)
{
    _check();

    auto status = (*_interface.prnnDestroyTensorDescriptor)(tensorDesc);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnDestroyTensorDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnCreateRNNDescriptor(prnnRNNDescriptor_t* rnnDesc)
{
    _check();

    auto status = (*_interface.prnnCreateRNNDescriptor)(rnnDesc);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnCreateRnnDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnDestroyRNNDescriptor(prnnRNNDescriptor_t rnnDesc)
{
    _check();

    auto status = (*_interface.prnnDestroyRNNDescriptor)(rnnDesc);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnDestroyRnnDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnSetRNNDescriptor(prnnRNNDescriptor_t rnnDesc,
                                  int hiddenSize,
                                  int numLayers,
                                  prnnDropoutDescriptor_t dropoutDesc,
                                  prnnRNNInputMode_t inputMode,
                                  prnnDirectionMode_t direction,
                                  prnnRNNMode_t mode,
                                  prnnDataType_t dataType,
                                  prnnBackend_t backend)
{
    _check();

    auto status = (*_interface.prnnSetRNNDescriptor)(rnnDesc,
                                                     hiddenSize,
                                                     numLayers,
                                                     dropoutDesc,
                                                     inputMode,
                                                     direction,
                                                     mode,
                                                     dataType,
                                                     backend);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnSetRnnDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnGetRNNWorkspaceSize(const prnnRNNDescriptor_t rnnDesc,
                                          const int seqLength,
                                          const prnnTensorDescriptor_t* xDesc,
                                          size_t* sizeInBytes
                                          )
{
    _check();

    auto status = (*_interface.prnnGetRNNWorkspaceSize)(_interface.getHandle(),
                                                        rnnDesc,
                                                        seqLength,
                                                        xDesc,
                                                        sizeInBytes
                                                        );

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnGetRNNWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnGetRNNTrainingReserveSize(const prnnRNNDescriptor_t rnnDesc,
                                                const int seqLength,
                                                const prnnTensorDescriptor_t* xDesc,
                                                size_t* sizeInBytes
                                                )
{
    _check();

    auto status = (*_interface.prnnGetRNNWorkspaceSize)(_interface.getHandle(),
                                                        rnnDesc,
                                                        seqLength,
                                                        xDesc,
                                                        sizeInBytes
                                                        );

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnGetRNNWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnGetRNNParamsSize(const prnnRNNDescriptor_t rnnDesc,
                                       const prnnTensorDescriptor_t xDesc,
                                       size_t* sizeInBytes
                                       )
{
    _check();

    auto status = (*_interface.prnnGetRNNParamsSize)(_interface.getHandle(),
                                                     rnnDesc,
                                                     xDesc,
                                                     sizeInBytes);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnGetRNNParamsSize failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnGetRNNLinLayerMatrixParams(const prnnRNNDescriptor_t rnnDesc,
                                                 const int layer,
                                                 const prnnTensorDescriptor_t xDesc,
                                                 const prnnFilterDescriptor_t wDesc,
                                                 const void* w,
                                                 const int linLayerID,
                                                 prnnFilterDescriptor_t linLayerMatDesc,
                                                 void** linLayerMat
                                                 )
{
    _check();

    auto status = (*_interface.prnnGetRNNLinLayerMatrixParams)(_interface.getHandle(),
                                                               rnnDesc,
                                                               layer,
                                                               xDesc,
                                                               wDesc,
                                                               w,
                                                               linLayerID,
                                                               linLayerMatDesc,
                                                               linLayerMat);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnGetRNNParamsSize failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnGetRNNLinLayerBiasParams(const prnnRNNDescriptor_t rnnDesc,
                                               const int layer,
                                               const prnnTensorDescriptor_t xDesc,
                                               const prnnFilterDescriptor_t wDesc,
                                               const void* w,
                                               const int linLayerID,
                                               prnnFilterDescriptor_t linLayerBiasDesc,
                                               void** linLayerBias
                                               )
{
    _check();

    auto status = (*_interface.prnnGetRNNLinLayerBiasParams)(_interface.getHandle(),
                                                             rnnDesc,
                                                             layer,
                                                             xDesc,
                                                             wDesc,
                                                             w,
                                                             linLayerID,
                                                             linLayerBiasDesc,
                                                             linLayerBias);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnGetRNNLinLayerBiasParams failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnRNNForward(const prnnRNNDescriptor_t rnnDesc,
                                 const int seqLength,
                                 const prnnTensorDescriptor_t* xDesc,
                                 const void* x,
                                 const prnnTensorDescriptor_t hxDesc,
                                 const void* hx,
                                 const prnnTensorDescriptor_t cxDesc,
                                 const void* cx,
                                 const prnnFilterDescriptor_t wDesc,
                                 const void* w,
                                 const prnnTensorDescriptor_t* yDesc,
                                 void* y,
                                 const prnnTensorDescriptor_t hyDesc,
                                 void* hy,
                                 const prnnTensorDescriptor_t cyDesc,
                                 void* cy,
                                 void* workspace,
                                 size_t workSpaceSizeInBytes,
                                 void* reserveSpace,
                                 size_t reserveSpaceSizeInBytes)
{
    _check();

    auto status = (*_interface.prnnRNNForward)(_interface.getHandle(),
                                               rnnDesc,
                                               seqLength,
                                               xDesc,
                                               x,
                                               hxDesc,
                                               hx,
                                               cxDesc,
                                               cx,
                                               wDesc,
                                               w,
                                               yDesc,
                                               y,
                                               hyDesc,
                                               hy,
                                               cyDesc,
                                               cy,
                                               workspace,
                                               workSpaceSizeInBytes,
                                               reserveSpace,
                                               reserveSpaceSizeInBytes);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnRNNForward failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnRNNBackwardData(const prnnRNNDescriptor_t rnnDesc,
                                      const int seqLength,
                                      const prnnTensorDescriptor_t* yDesc,
                                      const void* y,
                                      const prnnTensorDescriptor_t* dyDesc,
                                      const void* dy,
                                      const prnnTensorDescriptor_t dhyDesc,
                                      const void* dhy,
                                      const prnnTensorDescriptor_t dcyDesc,
                                      const void* dcy,
                                      const prnnFilterDescriptor_t wDesc,
                                      const void* w,
                                      const prnnTensorDescriptor_t hxDesc,
                                      const void* hx,
                                      const prnnTensorDescriptor_t cxDesc,
                                      const void* cx,
                                      const prnnTensorDescriptor_t* dxDesc,
                                      void* dx,
                                      const prnnTensorDescriptor_t dhxDesc,
                                      void* dhx,
                                      const prnnTensorDescriptor_t dcxDesc,
                                      void* dcx,
                                      void* workspace,
                                      size_t workSpaceSizeInBytes,
                                      void* reserveSpace,
                                      size_t reserveSpaceSizeInBytes)
{
    _check();

    auto status = (*_interface.prnnRNNBackwardData)(_interface.getHandle(),
                                                    rnnDesc,
                                                    seqLength,
                                                    yDesc,
                                                    y,
                                                    dyDesc,
                                                    dy,
                                                    dhyDesc,
                                                    dhy,
                                                    dcyDesc,
                                                    dcy,
                                                    wDesc,
                                                    w,
                                                    hxDesc,
                                                    hx,
                                                    cxDesc,
                                                    cx,
                                                    dxDesc,
                                                    dx,
                                                    dhxDesc,
                                                    dhx,
                                                    dcxDesc,
                                                    dcx,
                                                    workspace,
                                                    workSpaceSizeInBytes,
                                                    reserveSpace,
                                                    reserveSpaceSizeInBytes);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnRNNBackwardData failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::prnnRNNBackwardWeights(const prnnRNNDescriptor_t rnnDesc,
                                         const int seqLength,
                                         const prnnTensorDescriptor_t* xDesc,
                                         const void* x,
                                         const prnnTensorDescriptor_t hxDesc,
                                         const void* hx,
                                         const prnnTensorDescriptor_t* yDesc,
                                         const void* y,
                                         const void* workspace,
                                         size_t workSpaceSizeInBytes,
                                         const prnnFilterDescriptor_t dwDesc,
                                         void* dw,
                                         const void* reserveSpace,
                                         size_t reserveSpaceSizeInBytes)
{
    _check();

    auto status = (*_interface.prnnRNNBackwardWeights)(_interface.getHandle(),
                                                       rnnDesc,
                                                       seqLength,
                                                       xDesc,
                                                       x,
                                                       hxDesc,
                                                       hx,
                                                       yDesc,
                                                       y,
                                                       workspace,
                                                       workSpaceSizeInBytes,
                                                       dwDesc,
                                                       dw,
                                                       reserveSpace,
                                                       reserveSpaceSizeInBytes);

    if(status != PRNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("prnnRNNBackwardWeights failed: " +
            _interface.getErrorString(status));
    }
}

void PrnnLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call PRNN function when "
            "the library is not loaded. Loading library failed, consider "
            "installing PRNN.");
    }
}

static void checkFunction(void* pointer, const std::string& name)
{
    if(pointer == nullptr)
    {
        throw std::runtime_error("Failed to load function '" + name +
            "' from dynamic library.");
    }
}

PrnnLibrary::Interface::Interface()
: _library(nullptr), _failed(false), _handle(nullptr)
{

}

PrnnLibrary::Interface::~Interface()
{
    unload();
}

void PrnnLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;
    if(!parallel::isCudaEnabled()) return;

    #ifdef __APPLE__
    const char* libraryName = "libprnn.dylib";
    #else
    const char* libraryName = "libprnn.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("PrnnLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("PrnnLibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    try
    {
        #define DynLink( function ) \
            util::bit_cast(function, dlsym(_library, #function)); \
            checkFunction((void*)function, #function)

        DynLink(prnnGetErrorString);

        DynLink(prnnCreate);
        DynLink(prnnDestroy);
        DynLink(prnnSetStream);
        DynLink(prnnGetStream);

        DynLink(prnnCreateTensorDescriptor);
        DynLink(prnnSetTensorNdDescriptor);
        DynLink(prnnGetTensorNdDescriptor);
        DynLink(prnnDestroyTensorDescriptor);

        DynLink(prnnCreateRNNDescriptor);
        DynLink(prnnDestroyRNNDescriptor);
        DynLink(prnnSetRNNDescriptor);

        DynLink(prnnGetRNNWorkspaceSize);
        DynLink(prnnGetRNNTrainingReserveSize);
        DynLink(prnnGetRNNParamsSize);
        DynLink(prnnGetRNNLinLayerMatrixParams);
        DynLink(prnnGetRNNLinLayerBiasParams);

        DynLink(prnnRNNForward);
        DynLink(prnnRNNBackwardData);
        DynLink(prnnRNNBackwardWeights);

        #undef DynLink

        auto status = (*prnnCreate)(&_handle);

        if(status != PRNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("prnnCreate failed: " + getErrorString(status));
        }

        util::log("PrnnLibrary") << " Loaded library '" << libraryName
            << "' successfully\n";
    }
    catch(...)
    {
        unload();
        throw;
    }

}

bool PrnnLibrary::Interface::loaded() const
{
    return !_failed && (_library != nullptr);
}

void PrnnLibrary::Interface::unload()
{
    if (!loaded()) return;

    dlclose(_library);

    _library = nullptr;
}

PrnnLibrary::prnnHandle_t PrnnLibrary::Interface::getHandle()
{
    return _handle;
}

std::string PrnnLibrary::Interface::getErrorString(prnnStatus_t status)
{
    _check();

    return (*prnnGetErrorString)(status);
}

PrnnLibrary::Interface PrnnLibrary::_interface;

}

}





