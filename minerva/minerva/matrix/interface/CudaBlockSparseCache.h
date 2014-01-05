/*! \file   CudaBlockSparseCache.h
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The header file for the CudaBlockSparseCache class.
*/

#pragma once

// Standard Library Includes
#include <cstdlib>

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrixImplementation; } }

namespace minerva
{

namespace matrix
{

class CudaBlockSparseCache
{
public:
	static float* acquire(const BlockSparseMatrixImplementation* matrix) ;
	static float* acquireReadOnly(const BlockSparseMatrixImplementation* matrix);
	static float* acquireClobber(const BlockSparseMatrixImplementation* matrix);
	static void release(const BlockSparseMatrixImplementation* matrix);

public:
	static void invalidate(const BlockSparseMatrixImplementation* matrix);
	static void synchronize(const BlockSparseMatrixImplementation* matrix);
	static void synchronizeHostReadOnly(const BlockSparseMatrixImplementation* matrix);

public:
	static void* fastDeviceAllocate(size_t size);
	static void fastDeviceFree(void*);

};

}

}

