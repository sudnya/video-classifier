/*! \file   CudaBlockSparseCache.h
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The header file for the CudaBlockSparseCache class.
*/

#pragma once

// Forward Declarations
namespace minerva { namespace matrix { class BlockSparseMatrixImplementation; } }

namespace minerva
{

namespace matrix
{

class CudaBlockSparseCache
{
public:
	float* acquire(BlockSparseMatrixImplementation* matrix) const;
	float* acquireReadyOnly(BlockSparseMatrixImplementation* matrix) const;
	float* acquireClobber(BlockSparseMatrixImplementation* matrix) const;
	void release(BlockSparseMatrixImplementation* matrix) const;

public:
	void invalidate(BlockSparseMatrixImplementation* matrix) const;
	void synchronize(BlockSparseMatrixImplementation* matrix) const;

};

}

}

