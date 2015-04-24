/*    \file   CudnnLibrary.cpp
    \date   April 23, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the CudnnLibrary class.
*/


// Standard Library Includes
#include <minerva/matrix/interface/CudnnLibrary.h>
#include <cassert>

namespace minerva
{

namespace matrix
{

void CudnnLibrary::CudnnLibrary::load(){ assert(false && "Not implemented");}
bool CudnnLibrary::loaded(){ assert(false && "Not implemented");}

/* Create an instance of a generic Tensor descriptor */
void CudnnLibrary::cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t   tensorDesc,
                                   cudnnTensorFormat_t  format,
                                   cudnnDataType_t dataType, // image data type
                                   int n,        // number of inputs (batch size)
                                   int c,        // number of input feature maps
                                   int h,        // height of input section
                                   int w         // width of input section
                                   ){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnTransformTensor(const void*                      alpha,
                             const cudnnTensorDescriptor_t    srcDesc,
                             const void*                      srcData,
                             const void*                      beta,
                             const cudnnTensorDescriptor_t    destDesc,
                             void*                            destData){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                   cudnnDataType_t dataType, // image data type
                                   int k,        // number of output feature maps
                                   int c,        // number of input feature maps
                                   int h,        // height of each input filter
                                   int w         // width of  each input fitler
                                   ){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                        int pad_h,    // zero-padding height
                                        int pad_w,    // zero-padding width
                                        int u,        // vertical filter stride
                                        int v,        // horizontal filter stride
                                        int upscalex, // upscale the input in x-direction
                                        int upscaley, // upscale the input in y-direction
                                        cudnnConvolutionMode_t mode){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc){ assert(false && "Not implemented");}


void CudnnLibrary::cudnnGetConvolutionForwardAlgorithm(const cudnnTensorDescriptor_t      srcDesc,
                                            const cudnnFilterDescriptor_t      filterDesc,
                                            const cudnnConvolutionDescriptor_t convDesc,
                                            const cudnnTensorDescriptor_t      destDesc,
                                            cudnnConvolutionFwdPreference_t    preference,
                                            size_t                             memoryLimitInbytes,
                                            cudnnConvolutionFwdAlgo_t*         algo){ assert(false && "Not implemented");}

void CudnnLibrary::cudnnGetConvolutionForwardWorkspaceSize(const cudnnTensorDescriptor_t      srcDesc,
                                                const cudnnFilterDescriptor_t      filterDesc,
                                                const cudnnConvolutionDescriptor_t convDesc,
                                                const cudnnTensorDescriptor_t      destDesc,
                                                cudnnConvolutionFwdAlgo_t          algo,
                                                size_t*                            sizeInBytes){ assert(false && "Not implemented");}


void CudnnLibrary::cudnnConvolutionForward(const void*                        alpha,
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
                                void*                              destData){ assert(false && "Not implemented");}




void CudnnLibrary::_check(){ assert(false && "Not implemented");}


/*! \brief The constructor zeros out all of the pointers */
CudnnLibrary::Interface::Interface() : _library(nullptr), _failed(false), _handle(0) {

    assert(false && "Not implemented");
}

/*! \brief The destructor closes dlls */
CudnnLibrary::Interface::~Interface(){ assert(false && "Not implemented");}
/*! \brief Load the library */
void CudnnLibrary::Interface::load(){ assert(false && "Not implemented");}
/*! \brief Has the library been loaded? */
bool CudnnLibrary::Interface::loaded() const { return !_failed && (_library != nullptr) && (_handle != 0);}
/*! \brief unloads the library */
void CudnnLibrary::Interface::unload(){ assert(false && "Not implemented");}



}

}




