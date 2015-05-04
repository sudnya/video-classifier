/*  \file   CudnnLibrary.h
    \date   April 22, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CudnnLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>
#include <string>

namespace minerva
{

namespace matrix
{

/*! \brief A singleton interface to CUDNN if it is dynamically loaded. */
class CudnnLibrary
{
public:
    typedef enum
    {
        CUDNN_STATUS_SUCCESS          = 0,
        CUDNN_STATUS_NOT_INITIALIZED  = 1,
        CUDNN_STATUS_ALLOC_FAILED     = 2,
        CUDNN_STATUS_BAD_PARAM        = 3,
        CUDNN_STATUS_INTERNAL_ERROR   = 4,
        CUDNN_STATUS_INVALID_VALUE    = 5,
        CUDNN_STATUS_ARCH_MISMATCH    = 6,
        CUDNN_STATUS_MAPPING_ERROR    = 7,
        CUDNN_STATUS_EXECUTION_FAILED = 8,
        CUDNN_STATUS_NOT_SUPPORTED    = 9,
        CUDNN_STATUS_LICENSE_ERROR    = 10
    } cudnnStatus_t;

    /* Data structures to represent Image/Filter and the Neural Network Layer */
    typedef struct cudnnTensorStruct*        cudnnTensorDescriptor_t;
    typedef struct cudnnConvolutionStruct*   cudnnConvolutionDescriptor_t;
    typedef struct cudnnPoolingStruct*       cudnnPoolingDescriptor_t;
    typedef struct cudnnFilterStruct*        cudnnFilterDescriptor_t;

    /* CUDNN data type */
    typedef enum
    {
        CUDNN_DATA_FLOAT  = 0,
        CUDNN_DATA_DOUBLE = 1
    } cudnnDataType_t;

    typedef enum
    {
        CUDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
        CUDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
    } cudnnTensorFormat_t;

    /* convolution mode */
    typedef enum
    {
        CUDNN_CONVOLUTION       = 0,
        CUDNN_CROSS_CORRELATION = 1
    } cudnnConvolutionMode_t;

    typedef enum
    {
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE        = 0,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST      = 1,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
    } cudnnConvolutionFwdPreference_t;

    typedef enum
    {
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3
    } cudnnConvolutionFwdAlgo_t;


public:
    static void load();
    static bool loaded();

public:
    /* Create an instance of a generic Tensor descriptor */
    static void cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);

    static void cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t   tensorDesc,
                                           cudnnTensorFormat_t  format,
                                           cudnnDataType_t dataType, // image data type
                                           int n,        // number of inputs (batch size)
                                           int c,        // number of input feature maps
                                           int h,        // height of input section
                                           int w         // width of input section
                                           );

    static void cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);

    static void cudnnTransformTensor(const void*                      alpha,
                                     const cudnnTensorDescriptor_t    srcDesc,
                                     const void*                      srcData,
                                     const void*                      beta,
                                     const cudnnTensorDescriptor_t    destDesc,
                                     void*                            destData);

public:
    static void cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc);

    static void cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                           cudnnDataType_t dataType, // image data type
                                           int k,        // number of output feature maps
                                           int c,        // number of input feature maps
                                           int h,        // height of each input filter
                                           int w         // width of  each input fitler
                                           );

    static void cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);

public:
    static void cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);

    static void cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                                int pad_h,    // zero-padding height
                                                int pad_w,    // zero-padding width
                                                int u,        // vertical filter stride
                                                int v,        // horizontal filter stride
                                                int upscalex, // upscale the input in x-direction
                                                int upscaley, // upscale the input in y-direction
                                                cudnnConvolutionMode_t mode);

    static void cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);


public:
    static void cudnnGetConvolutionForwardAlgorithm(const cudnnTensorDescriptor_t      srcDesc,
                                                    const cudnnFilterDescriptor_t      filterDesc,
                                                    const cudnnConvolutionDescriptor_t convDesc,
                                                    const cudnnTensorDescriptor_t      destDesc,
                                                    cudnnConvolutionFwdPreference_t    preference,
                                                    size_t                             memoryLimitInbytes,
                                                    cudnnConvolutionFwdAlgo_t*         algo);

    static void cudnnGetConvolutionForwardWorkspaceSize(const cudnnTensorDescriptor_t      srcDesc,
                                                        const cudnnFilterDescriptor_t      filterDesc,
                                                        const cudnnConvolutionDescriptor_t convDesc,
                                                        const cudnnTensorDescriptor_t      destDesc,
                                                        cudnnConvolutionFwdAlgo_t          algo,
                                                        size_t*                            sizeInBytes);


    static void cudnnConvolutionForward(const void*                        alpha,
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
                                        void*                              destData);

public:
    static void cudnnConvolutionBackwardData(const void*                         alpha,
                                             const cudnnFilterDescriptor_t       filterDesc,
                                             const void                         *filterData,
                                             const cudnnTensorDescriptor_t       diffDesc,
                                             const void*                         diffData,
                                             const cudnnConvolutionDescriptor_t  convDesc,
                                             const void*                         beta,
                                             const cudnnTensorDescriptor_t       gradDesc,
                                             void*                               gradData);




private:
    static void _check();

private:
    typedef struct cudnnContext* cudnnHandle_t;

    class Interface
    {
    public:
        const char* (*cudnnGetErrorString)(cudnnStatus_t status);

        cudnnStatus_t (*cudnnCreate)(cudnnHandle_t* handle);
        cudnnStatus_t (*cudnnDestroy)(cudnnHandle_t handle);

    public:
        /* Create an instance of a generic Tensor descriptor */
        cudnnStatus_t (*cudnnCreateTensorDescriptor)(cudnnTensorDescriptor_t* tensorDesc);

        cudnnStatus_t (*cudnnSetTensor4dDescriptor)(cudnnTensorDescriptor_t   tensorDesc,
                                            cudnnTensorFormat_t  format,
                                            cudnnDataType_t dataType, // image data type
                                            int n,        // number of inputs (batch size)
                                            int c,        // number of input feature maps
                                            int h,        // height of input section
                                            int w         // width of input section
                                        );

        cudnnStatus_t (*cudnnDestroyTensorDescriptor)( cudnnTensorDescriptor_t tensorDesc );

        cudnnStatus_t (*cudnnTransformTensor)(cudnnHandle_t                    handle,
                                              const void*                      alpha,
                                              const cudnnTensorDescriptor_t    srcDesc,
                                              const void*                      srcData,
                                              const void*                      beta,
                                              const cudnnTensorDescriptor_t    destDesc,
                                              void*                            destData);

    public:
        cudnnStatus_t (*cudnnCreateFilterDescriptor)(cudnnFilterDescriptor_t* filterDesc);

        cudnnStatus_t (*cudnnSetFilter4dDescriptor)(cudnnFilterDescriptor_t filterDesc,
                                                    cudnnDataType_t dataType, // image data type
                                                    int k,        // number of output feature maps
                                                    int c,        // number of input feature maps
                                                    int h,        // height of each input filter
                                                    int w         // width of  each input fitler
                                                      );

        cudnnStatus_t (*cudnnDestroyFilterDescriptor)(cudnnFilterDescriptor_t filterDesc);

    public:
        cudnnStatus_t (*cudnnCreateConvolutionDescriptor)(cudnnConvolutionDescriptor_t* convDesc);

        cudnnStatus_t (*cudnnSetConvolution2dDescriptor)(cudnnConvolutionDescriptor_t convDesc,
                                                         int pad_h,    // zero-padding height
                                                         int pad_w,    // zero-padding width
                                                         int u,        // vertical filter stride
                                                         int v,        // horizontal filter stride
                                                         int upscalex, // upscale the input in x-direction
                                                         int upscaley, // upscale the input in y-direction
                                                         cudnnConvolutionMode_t mode);

        cudnnStatus_t (*cudnnDestroyConvolutionDescriptor)(cudnnConvolutionDescriptor_t convDesc);


    public:
        cudnnStatus_t (*cudnnGetConvolutionForwardAlgorithm)(cudnnHandle_t                      handle,
                                                             const cudnnTensorDescriptor_t      srcDesc,
                                                             const cudnnFilterDescriptor_t      filterDesc,
                                                             const cudnnConvolutionDescriptor_t convDesc,
                                                             const cudnnTensorDescriptor_t      destDesc,
                                                             cudnnConvolutionFwdPreference_t    preference,
                                                             size_t                             memoryLimitInbytes,
                                                             cudnnConvolutionFwdAlgo_t*         algo);

        cudnnStatus_t (*cudnnGetConvolutionForwardWorkspaceSize)(cudnnHandle_t                      handle,
                                                                 const cudnnTensorDescriptor_t      srcDesc,
                                                                 const cudnnFilterDescriptor_t      filterDesc,
                                                                 const cudnnConvolutionDescriptor_t convDesc,
                                                                 const cudnnTensorDescriptor_t      destDesc,
                                                                 cudnnConvolutionFwdAlgo_t          algo,
                                                                 size_t*                            sizeInBytes);


        cudnnStatus_t (*cudnnConvolutionForward)(cudnnHandle_t                       handle,
                                                 const void*                         alpha,
                                                 const cudnnTensorDescriptor_t       srcDesc,
                                                 const void*                         srcData,
                                                 const cudnnFilterDescriptor_t       filterDesc,
                                                 const void*                         filterData,
                                                 const cudnnConvolutionDescriptor_t  convDesc,
                                                 cudnnConvolutionFwdAlgo_t           algo,
                                                 void*                               workSpace,
                                                 size_t                              workSpaceSizeInBytes,
                                                 const void*                         beta,
                                                 const cudnnTensorDescriptor_t       destDesc,
                                                 void*                               destData);

        cudnnStatus_t (*cudnnConvolutionBackwardData)(cudnnHandle_t                  handle,
                                                 const void*                         alpha,
                                                 const cudnnFilterDescriptor_t       filterDesc,
                                                 const void                         *filterData,
                                                 const cudnnTensorDescriptor_t       diffDesc,
                                                 const void*                         diffData,
                                                 const cudnnConvolutionDescriptor_t  convDesc,
                                                 const void*                         beta,
                                                 const cudnnTensorDescriptor_t       gradDesc,
                                                 void*                               gradData);

    public:
        /*! \brief The constructor zeros out all of the pointers */
        Interface();

        /*! \brief The destructor closes dlls */
        ~Interface();
        /*! \brief Load the library */
        void load();
        /*! \brief Has the library been loaded? */
        bool loaded() const;
        /*! \brief unloads the library */
        void unload();

    public:
        cudnnHandle_t getHandle();

    public:
        std::string getErrorString(cudnnStatus_t status);

    private:
        void* _library;
        bool  _failed;

    private:
        cudnnHandle_t _handle;
    };

private:
    static Interface _interface;

};

}

}



