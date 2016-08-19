/*  \file   PrnnLibrary.h
    \date   July 31, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the PrnnLibrary class.
*/

#pragma once

// Standard Library Includes
#include <cstddef>
#include <string>

// Forward Declarations
typedef struct prnnTensorStruct*  prnnTensorDescriptor_t;
typedef struct prnnTensorStruct*  prnnFilterDescriptor_t;
typedef struct prnnDropoutStruct* prnnDropoutDescriptor_t;
typedef struct prnnRNNStruct*     prnnRNNDescriptor_t;

namespace lucius
{

namespace matrix
{

/*! \brief A singleton interface to PRNN if it is dynamically loaded. */
class PrnnLibrary
{
public:
    typedef enum
    {
        PRNN_STATUS_SUCCESS          = 0,
        PRNN_STATUS_NOT_INITIALIZED  = 1,
        PRNN_STATUS_ALLOC_FAILED     = 2,
        PRNN_STATUS_BAD_PARAM        = 3,
        PRNN_STATUS_INTERNAL_ERROR   = 4,
        PRNN_STATUS_INVALID_VALUE    = 5,
        PRNN_STATUS_ARCH_MISMATCH    = 6,
        PRNN_STATUS_MAPPING_ERROR    = 7,
        PRNN_STATUS_EXECUTION_FAILED = 8,
        PRNN_STATUS_NOT_SUPPORTED    = 9
    } prnnStatus_t;

    typedef enum
    {
        PRNN_DATA_FLOAT   = 0,
        PRNN_DATA_DOUBLE  = 1,
        PRNN_DATA_HALF    = 2,
        PRNN_INVALID_DATA = 3,
    } prnnDataType_t;

    typedef enum
    {
        PRNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
        PRNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
    } prnnTensorFormat_t;

    typedef enum
    {
        PRNN_RNN_RELU = 0, // Stock RNN with ReLu activation
        PRNN_RNN_TANH = 1, // Stock RNN with tanh activation
        PRNN_LSTM     = 2, // LSTM with no peephole connections
        PRNN_GRU      = 3  // Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);
    } prnnRNNMode_t;

    typedef enum
    {
       PRNN_UNIDIRECTIONAL = 0,
       PRNN_BIDIRECTIONAL  = 1,  // Using output concatination at each step. Do we also want to support output sum?
       PRNN_REVERSE        = 2
    } prnnDirectionMode_t;

    typedef enum
    {
       PRNN_LINEAR_INPUT = 0,
       PRNN_SKIP_INPUT   = 1
    } prnnRNNInputMode_t;

    typedef enum
    {
        PRNN_PERSISTENT_BACKEND = 0,
        PRNN_CUDNN_BACKEND = 1,
        PRNN_BEST_BACKEND = 2
    } prnnBackend_t;


public:
    static void load();
    static bool loaded();

public:
    static const char* prnnGetErrorString(prnnStatus_t status);

public:
    /* Create an instance of a generic Tensor descriptor */
    static void prnnCreateTensorDescriptor(prnnTensorDescriptor_t* tensorDesc);

    static void prnnSetTensorNdDescriptor(prnnTensorDescriptor_t tensorDesc,
                                          prnnDataType_t         dataType,
                                          int                    nbDims,
                                          const int*             dimA,
                                          const int*             strideA);

    static void prnnGetTensorNdDescriptor(const prnnTensorDescriptor_t tensorDesc,
                                          int                          nbDimsRequested,
                                          prnnDataType_t*              dataType,
                                          int*                         nbDims,
                                          int*                         dimA,
                                          int*                         strideA);

    /* Destroy an instance of Tensor4d descriptor */
    static void prnnDestroyTensorDescriptor(prnnTensorDescriptor_t tensorDesc);

public:
    static void prnnCreateRNNDescriptor(prnnRNNDescriptor_t* rnnDesc);
    static void prnnDestroyRNNDescriptor(prnnRNNDescriptor_t rnnDesc);

    static void prnnSetRNNDescriptor(prnnRNNDescriptor_t rnnDesc,
                                      int hiddenSize,
                                      int numLayers,
                                      prnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                      prnnRNNInputMode_t inputMode,
                                      prnnDirectionMode_t direction,
                                      prnnRNNMode_t mode,
                                      prnnDataType_t dataType,
                                      prnnBackend_t backend);

    // dataType in the RNN descriptor is used to determine math precision
    // dataType in weight descriptors and input descriptors is used to describe storage

    static void prnnGetRNNWorkspaceSize(const prnnRNNDescriptor_t rnnDesc,
                                        const int seqLength,
                                        const prnnTensorDescriptor_t* xDesc,
                                        size_t* sizeInBytes
                                        );

    static void prnnGetRNNTrainingReserveSize(const prnnRNNDescriptor_t rnnDesc,
                                              const int seqLength,
                                              const prnnTensorDescriptor_t* xDesc,
                                              size_t* sizeInBytes
                                              );


    static void prnnGetRNNParamsSize(const prnnRNNDescriptor_t rnnDesc,
                                     const prnnTensorDescriptor_t xDesc,
                                     size_t* sizeInBytes
                                     );

    static void prnnGetRNNLinLayerMatrixParams(const prnnRNNDescriptor_t rnnDesc,
                                               const int layer,
                                               const prnnTensorDescriptor_t xDesc,
                                               const prnnFilterDescriptor_t wDesc,
                                               const void* w,
                                               const int linLayerID,
                                               prnnFilterDescriptor_t linLayerMatDesc,
                                               void** linLayerMat
                                               );

    static void prnnGetRNNLinLayerBiasParams(const prnnRNNDescriptor_t rnnDesc,
                                             const int layer,
                                             const prnnTensorDescriptor_t xDesc,
                                             const prnnFilterDescriptor_t wDesc,
                                             const void* w,
                                             const int linLayerID,
                                             prnnFilterDescriptor_t linLayerBiasDesc,
                                             void** linLayerBias
                                             );


    static void prnnRNNForward(const prnnRNNDescriptor_t rnnDesc,
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
                               size_t reserveSpaceSizeInBytes);

    static void prnnRNNBackwardData(const prnnRNNDescriptor_t rnnDesc,
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
                                    size_t reserveSpaceSizeInBytes);


    static void prnnRNNBackwardWeights(const prnnRNNDescriptor_t rnnDesc,
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
                                       size_t reserveSpaceSizeInBytes);

private:
    static void _check();

private:
    typedef struct prnnContext* prnnHandle_t;

    class Interface
    {
    public:
        const char* (*prnnGetErrorString)(prnnStatus_t status);

    public:
        prnnStatus_t (*prnnCreate)    (prnnHandle_t* handle);
        prnnStatus_t (*prnnDestroy)   (prnnHandle_t handle);
        prnnStatus_t (*prnnSetStream) (prnnHandle_t handle, void* streamId);
        prnnStatus_t (*prnnGetStream) (prnnHandle_t handle, void** streamId);

    public:
        /* Create an instance of a generic Tensor descriptor */
        prnnStatus_t (*prnnCreateTensorDescriptor)(prnnTensorDescriptor_t* tensorDesc);

        prnnStatus_t (*prnnSetTensorNdDescriptor)(prnnTensorDescriptor_t tensorDesc,
                                               prnnDataType_t         dataType,
                                               int                    nbDims,
                                               const int*             dimA,
                                               const int*             strideA);

        prnnStatus_t (*prnnGetTensorNdDescriptor)(const prnnTensorDescriptor_t tensorDesc,
                                               int                          nbDimsRequested,
                                               prnnDataType_t*              dataType,
                                               int*                         nbDims,
                                               int*                         dimA,
                                               int*                         strideA);

        /* Destroy an instance of Tensor4d descriptor */
        prnnStatus_t (*prnnDestroyTensorDescriptor)(prnnTensorDescriptor_t tensorDesc);

    public:
        prnnStatus_t (*prnnCreateRNNDescriptor)(prnnRNNDescriptor_t* rnnDesc);
        prnnStatus_t (*prnnDestroyRNNDescriptor)(prnnRNNDescriptor_t rnnDesc);

        prnnStatus_t (*prnnSetRNNDescriptor)(prnnRNNDescriptor_t rnnDesc,
                                          int hiddenSize,
                                          int numLayers,
                                          prnnDropoutDescriptor_t dropoutDesc, // Between layers, not between recurrent steps.
                                          prnnRNNInputMode_t inputMode,
                                          prnnDirectionMode_t direction,
                                          prnnRNNMode_t mode,
                                          prnnDataType_t dataType,
                                          prnnBackend_t backend);

        // dataType in the RNN descriptor is used to determine math precision
        // dataType in weight descriptors and input descriptors is used to describe storage

        prnnStatus_t (*prnnGetRNNWorkspaceSize)(prnnHandle_t handle,
                                             const prnnRNNDescriptor_t rnnDesc,
                                             const int seqLength,
                                             const prnnTensorDescriptor_t* xDesc,
                                             size_t* sizeInBytes
                                             );

        prnnStatus_t (*prnnGetRNNTrainingReserveSize)(prnnHandle_t handle,
                                                   const prnnRNNDescriptor_t rnnDesc,
                                                   const int seqLength,
                                                   const prnnTensorDescriptor_t* xDesc,
                                                   size_t* sizeInBytes
                                                   );


        prnnStatus_t (*prnnGetRNNParamsSize)(prnnHandle_t handle,
                                          const prnnRNNDescriptor_t rnnDesc,
                                          const prnnTensorDescriptor_t xDesc,
                                          size_t* sizeInBytes
                                          );

        prnnStatus_t (*prnnGetRNNLinLayerMatrixParams)(prnnHandle_t handle,
                                                    const prnnRNNDescriptor_t rnnDesc,
                                                    const int layer,
                                                    const prnnTensorDescriptor_t xDesc,
                                                    const prnnFilterDescriptor_t wDesc,
                                                    const void* w,
                                                    const int linLayerID,
                                                    prnnFilterDescriptor_t linLayerMatDesc,
                                                    void** linLayerMat
                                                    );

        prnnStatus_t (*prnnGetRNNLinLayerBiasParams)(prnnHandle_t handle,
                                                  const prnnRNNDescriptor_t rnnDesc,
                                                  const int layer,
                                                  const prnnTensorDescriptor_t xDesc,
                                                  const prnnFilterDescriptor_t wDesc,
                                                  const void* w,
                                                  const int linLayerID,
                                                  prnnFilterDescriptor_t linLayerBiasDesc,
                                                  void** linLayerBias
                                                  );


        prnnStatus_t (*prnnRNNForward)(prnnHandle_t handle,
                                    const prnnRNNDescriptor_t rnnDesc,
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
                                    size_t reserveSpaceSizeInBytes);

        prnnStatus_t (*prnnRNNBackwardData)(prnnHandle_t handle,
                                         const prnnRNNDescriptor_t rnnDesc,
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
                                         size_t reserveSpaceSizeInBytes);


        prnnStatus_t (*prnnRNNBackwardWeights)(prnnHandle_t handle,
                                            const prnnRNNDescriptor_t rnnDesc,
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
                                            size_t reserveSpaceSizeInBytes);

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
        prnnHandle_t getHandle();

    public:
        std::string getErrorString(prnnStatus_t status);

    private:
        void* _library;
        bool  _failed;

    private:
        prnnHandle_t _handle;
    };

private:
    static Interface _interface;

};

}

}




