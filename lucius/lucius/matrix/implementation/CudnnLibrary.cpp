/*    \file   CudnnLibrary.cpp
    \date   April 23, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the CudnnLibrary class.
*/


// Standard Library Includes
#include <lucius/matrix/interface/CudnnLibrary.h>

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

void CudnnLibrary::CudnnLibrary::load()
{
    _interface.load();
}

bool CudnnLibrary::loaded()
{
    load();

    return _interface.loaded();
}

/* Create an instance of a generic Tensor descriptor */
void CudnnLibrary::cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateTensorDescriptor)(tensorDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateTensorDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t   tensorDesc,
                                   cudnnTensorFormat_t  format,
                                   cudnnDataType_t dataType, // image data type
                                   int n,        // number of inputs (batch size)
                                   int c,        // number of input feature maps
                                   int h,        // height of input section
                                   int w         // width of input section
                                   )
{
    _check();

    auto status = (*_interface.cudnnSetTensor4dDescriptor)(tensorDesc, format, dataType, n, c, h, w);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetTensor4dDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyTensorDescriptor)(tensorDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyTensorDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnTransformTensor(const void*                      alpha,
                             const cudnnTensorDescriptor_t    srcDesc,
                             const void*                      srcData,
                             const void*                      beta,
                             const cudnnTensorDescriptor_t    destDesc,
                             void*                            destData)
{
    _check();

    auto status = (*_interface.cudnnTransformTensor)(_interface.getHandle(), alpha, srcDesc, srcData, beta, destDesc, destData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnTransformTensor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateFilterDescriptor)(filterDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateFilterDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                   cudnnDataType_t dataType, // image data type
                                   int k,        // number of output feature maps
                                   int c,        // number of input feature maps
                                   int h,        // height of each input filter
                                   int w         // width of  each input fitler
                                   )
{
    _check();

    auto status = (*_interface.cudnnSetFilter4dDescriptor)(filterDesc, dataType, k, c, h, w);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetFilter4dDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyFilterDescriptor)(filterDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyFilterDescriptor failed: " + _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc)
{
    _check();

    auto status = (*_interface.cudnnCreateConvolutionDescriptor)(convDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreateConvolutionDescriptor failed: " + _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                        int pad_h,    // zero-padding height
                                        int pad_w,    // zero-padding width
                                        int u,        // vertical filter stride
                                        int v,        // horizontal filter stride
                                        int upscalex, // upscale the input in x-direction
                                        int upscaley, // upscale the input in y-direction
                                        cudnnConvolutionMode_t mode)
{
    _check();

    auto status = (*_interface.cudnnSetConvolution2dDescriptor)(convDesc, pad_h, pad_w, u, v, upscalex, upscaley, mode);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetConvolution2dDescriptor failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyConvolutionDescriptor)(convDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyConvolutionDescriptor failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc)
{
    _check();

    auto status = (*_interface.cudnnCreatePoolingDescriptor)(poolingDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnCreatePoolingDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                    cudnnPoolingMode_t mode,
                                                    int windowHeight,
                                                    int windowWidth,
                                                    int verticalPadding,
                                                    int horizontalPadding,
                                                    int verticalStride,
                                                    int horizontalStride
                                               )
{
    _check();

    auto status = (*_interface.cudnnSetPooling2dDescriptor)(
        poolingDesc,
        mode,
        windowHeight,
        windowWidth,
        verticalPadding,
        horizontalPadding,
        verticalStride,
        horizontalStride
        );

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnSetPooling2dDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc)
{
    _check();

    auto status = (*_interface.cudnnDestroyPoolingDescriptor)(poolingDesc);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnDestroyPoolingDescriptor failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                             const cudnnTensorDescriptor_t inputTensorDesc,
                                             int* outN,
                                             int* outC,
                                             int* outH,
                                             int* outW)
{
    _check();

    auto status = (*_interface.cudnnGetPooling2dForwardOutputDim)(poolingDesc,
        inputTensorDesc,
        outN,
        outC,
        outH,
        outW);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetPooling2dForwardOutputDim failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetConvolutionForwardAlgorithm(const cudnnTensorDescriptor_t      srcDesc,
                                            const cudnnFilterDescriptor_t      filterDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t      destDesc,
                                            cudnnConvolutionFwdPreference_t    preference,
                                            size_t                             memoryLimitInbytes,
                                            cudnnConvolutionFwdAlgo_t*         algo)
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionForwardAlgorithm)(_interface.getHandle(),
        srcDesc, filterDesc, convDesc,
        destDesc, preference, memoryLimitInbytes, algo);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetConvolutionForwardAlgorithm failed: " +
            _interface.getErrorString(status));
    }

}

void CudnnLibrary::cudnnGetConvolutionForwardWorkspaceSize(const cudnnTensorDescriptor_t srcDesc,
                                                const cudnnFilterDescriptor_t      filterDesc,
                                                const cudnnConvolutionDescriptor_t convDesc,
                                                const cudnnTensorDescriptor_t      destDesc,
                                                cudnnConvolutionFwdAlgo_t          algo,
                                                size_t*                            sizeInBytes)
{
    _check();

    auto status = (*_interface.cudnnGetConvolutionForwardWorkspaceSize)(_interface.getHandle(),
        srcDesc, filterDesc, convDesc, destDesc, algo, sizeInBytes);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnGetConvolutionForwardWorkspaceSize failed: " +
            _interface.getErrorString(status));
    }

}


void CudnnLibrary::cudnnConvolutionForward(const void* alpha,
                                const cudnnTensorDescriptor_t      srcDesc,
                                const void*                        srcData,
                                const cudnnFilterDescriptor_t      filterDesc,
                                const void*                        filterData,
                                const cudnnConvolutionDescriptor_t convDesc,
                                cudnnConvolutionFwdAlgo_t          algo,
                                void*                              workSpace,
                                size_t                             workSpaceSizeInBytes,
                                const void*                        beta,
                                const cudnnTensorDescriptor_t      destDesc,
                                void*                              destData)
{
    _check();

    auto status = (*_interface.cudnnConvolutionForward)(_interface.getHandle(),
        alpha, srcDesc, srcData, filterDesc, filterData, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, destDesc, destData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionForward failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnConvolutionBackwardData(const void*        alpha,
                                const cudnnFilterDescriptor_t      filterDesc,
                                const void*                        filterData,
                                const cudnnTensorDescriptor_t      diffDesc,
                                const void*                        diffData,
                                const cudnnConvolutionDescriptor_t convDesc,
                                const void*                        beta,
                                const cudnnTensorDescriptor_t      gradDesc,
                                void*                              gradData)
{
    _check();

    auto status = (*_interface.cudnnConvolutionBackwardData)(_interface.getHandle(),
        alpha, filterDesc, filterData, diffDesc, diffData, convDesc,
        beta, gradDesc, gradData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardData failed: " + _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnConvolutionBackwardFilter(const void*      alpha,
                                const cudnnTensorDescriptor_t      srcDesc,
                                const void*                        srcData,
                                const cudnnTensorDescriptor_t      diffDesc,
                                const void*                        diffData,
                                const cudnnConvolutionDescriptor_t convDesc,
                                const void*                        beta,
                                const cudnnFilterDescriptor_t      gradDesc,
                                void*                              gradData)
{
    _check();

    auto status = (*_interface.cudnnConvolutionBackwardFilter)(_interface.getHandle(),
        alpha, srcDesc, srcData, diffDesc, diffData, convDesc,
        beta, gradDesc, gradData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnConvolutionBackwardData failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnPoolingForward(const cudnnPoolingDescriptor_t   poolingDesc,
                                       const void*                      alpha,
                                       const cudnnTensorDescriptor_t    srcDesc,
                                       const void*                      srcData,
                                       const void*                      beta,
                                       const cudnnTensorDescriptor_t    destDesc,
                                       void*                            destData
                                             )
{
    _check();

    auto status = (*_interface.cudnnPoolingForward)(_interface.getHandle(),
        poolingDesc,
        alpha, srcDesc, srcData,
        beta, destDesc, destData);

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnPoolingForward failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::cudnnPoolingBackward(const cudnnPoolingDescriptor_t  poolingDesc,
                                        const void*                     alpha,
                                        const cudnnTensorDescriptor_t   srcDesc,
                                        const void*                     srcData,
                                        const cudnnTensorDescriptor_t   srcDiffDesc,
                                        const void*                     srcDiffData,
                                        const cudnnTensorDescriptor_t   destDesc,
                                        const void*                     destData,
                                        const void*                     beta,
                                        const cudnnTensorDescriptor_t   destDiffDesc,
                                        void*                           destDiffData
                                              )
{
    _check();

    auto status = (*_interface.cudnnPoolingBackward)(_interface.getHandle(),
        poolingDesc,
        alpha,
        srcDesc,
        srcData,
        srcDiffDesc,
        srcDiffData,
        destDesc,
        destData,
        beta,
        destDiffDesc,
        destDiffData
        );

    if(status != CUDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error("cudnnPoolingBackward failed: " +
            _interface.getErrorString(status));
    }
}

void CudnnLibrary::_check()
{
    load();

    if(!loaded())
    {
        throw std::runtime_error("Tried to call CUDNN function when "
            "the library is not loaded. Loading library failed, consider "
            "installing CUDNN.");
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

CudnnLibrary::Interface::Interface() : _library(nullptr), _failed(false), _handle(nullptr)
{

}

CudnnLibrary::Interface::~Interface()
{
    unload();
}

void CudnnLibrary::Interface::load()
{
    if(_failed)  return;
    if(loaded()) return;
    if(!parallel::isCudaEnabled()) return;

    #ifdef __APPLE__
    const char* libraryName = "libcudnn.dylib";
    #else
    const char* libraryName = "libcudnn.so";
    #endif

    _library = dlopen(libraryName, RTLD_LAZY);

    util::log("CudnnLibrary") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
    {
        util::log("Cudnnlibrary") << " Failed to load library '" << libraryName
            << "'\n";
        _failed = true;
        return;
    }

    try
    {
        #define DynLink( function ) \
            util::bit_cast(function, dlsym(_library, #function)); \
            checkFunction((void*)function, #function)

        DynLink(cudnnGetErrorString);
        DynLink(cudnnCreate);
        DynLink(cudnnDestroy);

        DynLink(cudnnCreateTensorDescriptor);
        DynLink(cudnnSetTensor4dDescriptor);
        DynLink(cudnnDestroyTensorDescriptor);
        DynLink(cudnnTransformTensor);

        DynLink(cudnnCreateFilterDescriptor);
        DynLink(cudnnSetFilter4dDescriptor);
        DynLink(cudnnDestroyFilterDescriptor);

        DynLink(cudnnCreateConvolutionDescriptor);
        DynLink(cudnnSetConvolution2dDescriptor);
        DynLink(cudnnDestroyConvolutionDescriptor);

        DynLink(cudnnCreatePoolingDescriptor);
        DynLink(cudnnSetPooling2dDescriptor);
        DynLink(cudnnDestroyPoolingDescriptor);
        DynLink(cudnnGetPooling2dForwardOutputDim);

        DynLink(cudnnGetConvolutionForwardAlgorithm);
        DynLink(cudnnGetConvolutionForwardWorkspaceSize);
        DynLink(cudnnConvolutionForward);

        DynLink(cudnnConvolutionBackwardData);

        DynLink(cudnnConvolutionBackwardFilter);

        DynLink(cudnnPoolingForward);
        DynLink(cudnnPoolingBackward);

        #undef DynLink

        auto status = (*cudnnCreate)(&_handle);

        if(status != CUDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("cudnnCreate failed: " + getErrorString(status));
        }

        util::log("Cudnnlibrary") << " Loaded library '" << libraryName
            << "' successfully\n";
    }
    catch(...)
    {
        unload();
        throw;
    }
}

bool CudnnLibrary::Interface::loaded() const
{
    return !_failed && (_library != nullptr);
}

void CudnnLibrary::Interface::unload()
{
    if(!loaded()) return;

    dlclose(_library);
    _library = nullptr;
}

CudnnLibrary::cudnnHandle_t CudnnLibrary::Interface::getHandle()
{
    return _handle;
}

std::string CudnnLibrary::Interface::getErrorString(cudnnStatus_t status)
{
    _check();

    return (*cudnnGetErrorString)(status);
}

CudnnLibrary::Interface CudnnLibrary::_interface;

}

}




