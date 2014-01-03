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
	float* acquire(const BlockSparseMatrixImplementation* matrix) const;
	float* acquireReadOnly(const BlockSparseMatrixImplementation* matrix) const;
	float* acquireClobber(const BlockSparseMatrixImplementation* matrix) const;
	void release(const BlockSparseMatrixImplementation* matrix) const;

public:
	void invalidate(const BlockSparseMatrixImplementation* matrix) const;
	void synchronize(const BlockSparseMatrixImplementation* matrix) const;
	void synchronizeHostReadOnly(const BlockSparseMatrixImplementation* matrix) const;

};

}

}

