/*! \file   CudaBlockSparseCache.cpp
	\author Gregory Diamos
	\date   Monday December 30, 2013
	\brief  The source file for the CudaBlockSparseCache class.
*/

// Minerva Includes 
#include <minerva/matrix/interface/CudaBlockSparseCache.h>
#include <minerva/matrix/interface/CudaBlockSparseMatrix.h>
#include <minerva/matrix/interface/CudaDriver.h>

#include <minerva/matrix/interface/Matrix.h>

#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <map>
#include <cassert>

namespace minerva
{

namespace matrix
{

class Allocation
{
public:
	Allocation(void* a, size_t s)
	: address(a), size(s), dirty(false)
	{

	}

public:
	void* address;
	size_t size;

public:
	bool dirty;
};

typedef std::map<const BlockSparseMatrixImplementation*, Allocation> AllocationMap;

class CacheManager
{
public:
	CacheManager()
	: totalSize(0), maximumSize(0)
	{
		
	}

	~CacheManager()
	{
	}

public:
	AllocationMap allocations;
	size_t totalSize;
	size_t maximumSize;

public:
	float* acquire(BlockSparseMatrixImplementation* matrix)
	{
		initialize();
		
		auto allocation = findExisting(matrix);
		
		if(allocation == nullptr)
		{
			allocation = createNewAllocation(matrix);
			
			cacheAllocation(matrix, allocation);
		}
	
		allocation->dirty = true;
	
		return reinterpret_cast<float*>(allocation->address);
	}

	float* acquireReadOnly(BlockSparseMatrixImplementation* matrix)
	{
		initialize();
		
		auto allocation = findExisting(matrix);
		
		if(allocation == nullptr)
		{
			allocation = createNewAllocation(matrix);
			
			cacheAllocation(matrix, allocation);
		}
	
		return reinterpret_cast<float*>(allocation->address);
	}

	float* acquireClobber(BlockSparseMatrixImplementation* matrix)
	{
		initialize();
		
		auto allocation = findExisting(matrix);
		
		if(allocation == nullptr)
		{
			allocation = createNewAllocation(matrix);
		}
		
		allocation->dirty = false;
		
		return reinterpret_cast<float*>(allocation->address);
	}
	
	void release(BlockSparseMatrixImplementation* matrix)
	{
		// Reclaim the memory immediately if the cache has exceeded the max size
		if(totalSize > maximumSize)
		{
			synchronize(matrix);
			invalidate(matrix);
		}
	}
	
public:
	void invalidate(BlockSparseMatrixImplementation* matrix)
	{
		// just free the allocation
		auto allocation = allocations.find(matrix);
		
		if(allocation != allocations.end())
		{
			util::log("CudaBlockSparseCache") << "Invalidating cache entry for matrix " << matrix << " ("
				<< matrix->rows() << " rows, " << matrix->columns()
				<< " columns) on the GPU at (" << allocation->second.address << ", "
				<< allocation->second.size << " bytes).\n";
			totalSize -= allocation->second.size;
			
			CudaDriver::cuMemFree((CUdeviceptr)allocation->second.address);

			allocations.erase(allocation);
		}
	}
	
	void synchronize(BlockSparseMatrixImplementation* matrix)
	{
		auto allocation = allocations.find(matrix);
		
		auto cudaMatrix = static_cast<CudaBlockSparseMatrix*>(matrix);
		
		if(allocation != allocations.end())
		{
			if(allocation->second.dirty)
			{
				// TODO: optimize this
				size_t position = 0;
				
				for(auto& block : cudaMatrix->rawData())
				{
					CudaDriver::cuMemcpyDtoH(block.data().data(),
						(CUdeviceptr)(position + (uint8_t*)allocation->second.address),
						sizeof(float) * block.size());
					
					position += sizeof(float) * block.size();
				}
				
				allocation->second.dirty = false;
			}
		}
	}

private:
	Allocation* createNewAllocation(BlockSparseMatrixImplementation* matrix)
	{
		size_t size = matrix->size() * sizeof(float);

		void* newMemory = nullptr;

		if(size > 0)
		{
			CudaDriver::cuMemAlloc((CUdeviceptr*)&newMemory, size);
		}

		auto newAllocation = allocations.insert(
			std::make_pair(matrix, Allocation(newMemory, size))).first;		
		
		util::log("CudaBlockSparseCache") << "Creating device allocation for matrix "
			<< matrix << " (" << matrix->rows() << " rows, " << matrix->columns()
			<< " columns) on the GPU at (" << newAllocation->second.address << ", "
			<< newAllocation->second.size << " bytes).\n";
	
		totalSize += size;
	
		return &newAllocation->second;
	}

	void cacheAllocation(BlockSparseMatrixImplementation* matrix, Allocation* allocation)
	{
		util::log("CudaBlockSparseCache") << "Caching matrix " << matrix << " ("
			<< matrix->rows() << " rows, " << matrix->columns()
			<< " columns) on the GPU at (" << allocation->address << ", "
			<< allocation->size << " bytes).\n";
		
		// TODO: optimize this
		size_t position = 0;
		
		auto cudaMatrix = static_cast<CudaBlockSparseMatrix*>(matrix);
		
		for(auto& block : cudaMatrix->rawData())
		{
			CudaDriver::cuMemcpyHtoD(
				(CUdeviceptr)(position + (uint8_t*)allocation->address),
				block.data().data(), sizeof(float) * block.size());
			
			position += sizeof(float) * block.size();
		}
		util::log("CudaBlockSparseCache") << " finished...\n";
	}

	Allocation* findExisting(BlockSparseMatrixImplementation* matrix)
	{
		auto allocation = allocations.find(matrix);

		if(allocation == allocations.end()) return nullptr;
		
		return &allocation->second;
	}

	void initialize()
	{
		if(maximumSize > 0) return;
		
		size_t percent = util::KnobDatabase::getKnobValue(
			"CudaBlockSparseCache::CachePercentOfGPUMemory", 25);
		
		size_t available = 0;
		size_t total     = 0;
		
		CudaDriver::cuMemGetInfo(&available, &total);
		
		maximumSize = total * percent / 100;
		
		util::log("CudaBlockSparseCache") << "Initializing cache, using up to "
			<< maximumSize << " bytes from GPU memory.\n";
	}


};

static std::unique_ptr<CacheManager> cacheManager(new CacheManager);

float* CudaBlockSparseCache::acquire(const BlockSparseMatrixImplementation* m) const
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->acquire(matrix);
}

float* CudaBlockSparseCache::acquireReadyOnly(const BlockSparseMatrixImplementation* m) const
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->acquireReadOnly(matrix);
}

float* CudaBlockSparseCache::acquireClobber(const BlockSparseMatrixImplementation* m) const
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->acquireClobber(matrix);
}

void CudaBlockSparseCache::release(const BlockSparseMatrixImplementation* m) const
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->release(matrix);
}

void CudaBlockSparseCache::invalidate(const BlockSparseMatrixImplementation* m) const
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->invalidate(matrix);
}

void CudaBlockSparseCache::synchronize(const BlockSparseMatrixImplementation* m) const
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->synchronize(matrix);
}

}

}


