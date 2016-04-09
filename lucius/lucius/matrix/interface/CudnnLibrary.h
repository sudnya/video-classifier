/*  \file   CudnnLibrary.h
    \date   April 22, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the CudnnLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>
#include <string>

// Forward Declarations
typedef struct cudnnTensorStruct*        cudnnTensorDescriptor_t;
typedef struct cudnnConvolutionStruct*   cudnnConvolutionDescriptor_t;
typedef struct cudnnPoolingStruct*       cudnnPoolingDescriptor_t;
typedef struct cudnnFilterStruct*        cudnnFilterDescriptor_t;

namespace lucius
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

    /* CUDNN data type */
    typedef enum
    {
        CUDNN_DATA_FLOAT  = 0,
        CUDNN_DATA_DOUBLE = 1,
        CUDNN_DATA_HALF   = 2
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

    typedef enum
    {
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE             = 0,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST           = 1,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = 2,
    } cudnnConvolutionBwdDataPreference_t;

    typedef enum
    {
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0          = 0, // non-deterministic
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1          = 1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT        = 2,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3
    } cudnnConvolutionBwdDataAlgo_t;

    typedef enum
    {
        CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2,
    } cudnnConvolutionBwdFilterPreference_t;

    typedef enum
    {
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = 1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = 2,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3         = 3   // non-deterministic, algo0 with workspace
    } cudnnConvolutionBwdFilterAlgo_t;

    typedef enum
    {
        CUDNN_POOLING_MAX     = 0,
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
    } cudnnPoolingMode_t;

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
    static void cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc);

    static void cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                                        cudnnPoolingMode_t mode,
                                                        int windowHeight,
                                                        int windowWidth,
                                                        int verticalPadding,
                                                        int horizontalPadding,
                                                        int verticalStride,
                                                        int horizontalStride
                                                   );

    static void cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);

    static void cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t poolingDesc,
                                                 const cudnnTensorDescriptor_t inputTensorDesc,
                                                 int* outN,
                                                 int* outC,
                                                 int* outH,
                                                 int* outW);



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
    static void cudnnGetConvolutionBackwardDataAlgorithm(
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataPreference_t preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionBwdDataAlgo_t*      algo );

    static void cudnnGetConvolutionBackwardDataWorkspaceSize(
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                size_t*                             sizeInBytes );

    static void cudnnConvolutionBackwardData(const void*                         alpha,
                                             const cudnnFilterDescriptor_t       filterDesc,
                                             const void*                         filterData,
                                             const cudnnTensorDescriptor_t       diffDesc,
                                             const void*                         diffData,
                                             const cudnnConvolutionDescriptor_t  convDesc,
                                             cudnnConvolutionBwdDataAlgo_t       algo,
                                             void*                               workSpace,
                                             size_t                              workSpaceSizeInBytes,
                                             const void*                         beta,
                                             const cudnnTensorDescriptor_t       gradDesc,
                                             void*                               gradData);

public:
    static void cudnnGetConvolutionBackwardFilterAlgorithm(
                            const cudnnTensorDescriptor_t         xDesc,
                            const cudnnTensorDescriptor_t         dyDesc,
                            const cudnnConvolutionDescriptor_t    convDesc,
                            const cudnnFilterDescriptor_t         dwDesc,
                            cudnnConvolutionBwdFilterPreference_t preference,
                            size_t                                memoryLimitInBytes,
                            cudnnConvolutionBwdFilterAlgo_t*      algo );

    static void cudnnGetConvolutionBackwardFilterWorkspaceSize(
                            const cudnnTensorDescriptor_t       xDesc,
                            const cudnnTensorDescriptor_t       dyDesc,
                            const cudnnConvolutionDescriptor_t  convDesc,
                            const cudnnFilterDescriptor_t       gradDesc,
                            cudnnConvolutionBwdFilterAlgo_t     algo,
                            size_t*                             sizeInBytes );

    static void cudnnConvolutionBackwardFilter(const void*                         alpha,
                                               const cudnnTensorDescriptor_t       srcDesc,
                                               const void*                         srcData,
                                               const cudnnTensorDescriptor_t       diffDesc,
                                               const void*                         diffData,
                                               const cudnnConvolutionDescriptor_t  convDesc,
                                               cudnnConvolutionBwdFilterAlgo_t     algo,
                                               void*                               workSpace,
                                               size_t                              workSpaceSizeInBytes,
                                               const void*                         beta,
                                               const cudnnFilterDescriptor_t       gradDesc,
                                               void*                               gradData);

public:
    static void cudnnPoolingForward(
                                        const cudnnPoolingDescriptor_t   poolingDesc,
                                        const void*                      alpha,
                                        const cudnnTensorDescriptor_t    srcDesc,
                                        const void*                      srcData,
                                        const void*                      beta,
                                        const cudnnTensorDescriptor_t    destDesc,
                                        void*                            destData
                                     );

    static void cudnnPoolingBackward(
                                       const cudnnPoolingDescriptor_t  poolingDesc,
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
                                     );

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
        cudnnStatus_t (*cudnnCreatePoolingDescriptor)(cudnnPoolingDescriptor_t* poolingDesc);

        cudnnStatus_t (*cudnnSetPooling2dDescriptor)(cudnnPoolingDescriptor_t poolingDesc,
                                                            cudnnPoolingMode_t mode,
                                                            int windowHeight,
                                                            int windowWidth,
                                                            int verticalPadding,
                                                            int horizontalPadding,
                                                            int verticalStride,
                                                            int horizontalStride
                                                       );

        cudnnStatus_t (*cudnnDestroyPoolingDescriptor)(cudnnPoolingDescriptor_t poolingDesc);

        cudnnStatus_t (*cudnnGetPooling2dForwardOutputDim)(const cudnnPoolingDescriptor_t poolingDesc,
                                                 const cudnnTensorDescriptor_t inputTensorDesc,
                                                 int* outN,
                                                 int* outC,
                                                 int* outH,
                                                 int* outW);


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

        cudnnStatus_t (*cudnnGetConvolutionBackwardDataAlgorithm)(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataPreference_t preference,
                                size_t                              memoryLimitInBytes,
                                cudnnConvolutionBwdDataAlgo_t*      algo );

        cudnnStatus_t (*cudnnGetConvolutionBackwardDataWorkspaceSize)(
                                cudnnHandle_t                       handle,
                                const cudnnFilterDescriptor_t       wDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnTensorDescriptor_t       dxDesc,
                                cudnnConvolutionBwdDataAlgo_t       algo,
                                size_t*                             sizeInBytes );

        cudnnStatus_t (*cudnnConvolutionBackwardData)(cudnnHandle_t                  handle,
                                                 const void*                         alpha,
                                                 const cudnnFilterDescriptor_t       filterDesc,
                                                 const void*                         filterData,
                                                 const cudnnTensorDescriptor_t       diffDesc,
                                                 const void*                         diffData,
                                                 const cudnnConvolutionDescriptor_t  convDesc,
                                                 cudnnConvolutionBwdDataAlgo_t       algo,
                                                 void*                               workSpace,
                                                 size_t                              workSpaceSizeInBytes,
                                                 const void*                         beta,
                                                 const cudnnTensorDescriptor_t       gradDesc,
                                                 void*                               gradData);

        cudnnStatus_t (*cudnnGetConvolutionBackwardFilterAlgorithm)(
                                cudnnHandle_t                         handle,
                                const cudnnTensorDescriptor_t         xDesc,
                                const cudnnTensorDescriptor_t         dyDesc,
                                const cudnnConvolutionDescriptor_t    convDesc,
                                const cudnnFilterDescriptor_t         dwDesc,
                                cudnnConvolutionBwdFilterPreference_t preference,
                                size_t                                memoryLimitInBytes,
                                cudnnConvolutionBwdFilterAlgo_t*      algo );

        cudnnStatus_t (*cudnnGetConvolutionBackwardFilterWorkspaceSize)(
                                cudnnHandle_t                       handle,
                                const cudnnTensorDescriptor_t       xDesc,
                                const cudnnTensorDescriptor_t       dyDesc,
                                const cudnnConvolutionDescriptor_t  convDesc,
                                const cudnnFilterDescriptor_t       gradDesc,
                                cudnnConvolutionBwdFilterAlgo_t     algo,
                                size_t*                             sizeInBytes );

        cudnnStatus_t (*cudnnConvolutionBackwardFilter)(cudnnHandle_t                handle,
                                                 const void*                         alpha,
                                                 const cudnnTensorDescriptor_t       srcDesc,
                                                 const void*                         srcData,
                                                 const cudnnTensorDescriptor_t       diffDesc,
                                                 const void*                         diffData,
                                                 const cudnnConvolutionDescriptor_t  convDesc,
                                                 cudnnConvolutionBwdFilterAlgo_t     algo,
                                                 void*                               workSpace,
                                                 size_t                              workSpaceSizeInBytes,
                                                 const void*                         beta,
                                                 const cudnnFilterDescriptor_t       gradDesc,
                                                 void*                               gradData);

    public:
        cudnnStatus_t (*cudnnPoolingForward)(cudnnHandle_t handle,
                                                    const cudnnPoolingDescriptor_t   poolingDesc,
                                                    const void*                      alpha,
                                                    const cudnnTensorDescriptor_t    srcDesc,
                                                    const void*                      srcData,
                                                    const void*                      beta,
                                                    const cudnnTensorDescriptor_t    destDesc,
                                                    void*                            destData
                                                 );

        cudnnStatus_t (*cudnnPoolingBackward)(cudnnHandle_t  handle,
                                                    const cudnnPoolingDescriptor_t  poolingDesc,
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
                                                  );

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



