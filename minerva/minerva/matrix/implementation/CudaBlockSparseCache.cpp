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

class PoolAllocation
{
public:
	PoolAllocation(void* a, size_t bytes)
	: address(a), size(bytes)
	{
		
	}

public:
	void* address;
	size_t size;

};

typedef std::map<void*, PoolAllocation> PoolAllocationMap;

class DeviceMemoryManager
{
public:
	DeviceMemoryManager()
	: isInitialized(false), memoryPoolBaseAddress(nullptr), memoryPoolSize(0)
	{
		
	}
	
	~DeviceMemoryManager()
	{
		//slowDeviceFree(memoryPoolBaseAddress);
	}
	
public:
	DeviceMemoryManager(const DeviceMemoryManager&) = delete;
	DeviceMemoryManager& operator=(const DeviceMemoryManager&) = delete;

public:
	void* fastDeviceAllocate(size_t bytes)
	{
		assert(isInitialized);
		
		#if 1
		auto allocation = tryFastAllocation(bytes);
		
		if(allocation != nullptr)
		{
			util::log("CudaBlockSparseCache") << " fast allocate of "
				<< allocation->address << " with " << bytes << " bytes\n";
			
			return allocation->address;
		}
		#endif
		
		return slowDeviceAllocate(bytes);
	}

	void fastDeviceFree(void* address)
	{
		assert(isInitialized);
		
		#if 1	
		if(address == nullptr) return;
		
		auto allocation = poolAllocations.find(address);
		
		if(allocation != poolAllocations.end())
		{
			util::log("CudaBlockSparseCache") << " fast free of "
				<< allocation->second.address << " with "
				<< allocation->second.size << " bytes\n";
			
			poolAllocations.erase(allocation);
			return;
		}
		#endif
		
		slowDeviceFree(address);
	}
	
	void initialize(size_t cacheSize)
	{
		assert(!isInitialized);
		
		isInitialized = true;
		
		memoryPoolBaseAddress = slowDeviceAllocate(cacheSize);
		
		memoryPoolSize = cacheSize;
	}

public:
	void* slowDeviceAllocate(size_t bytes)
	{
		void* address = nullptr;
		
		CudaDriver::cuMemAlloc((CUdeviceptr*)&address, bytes);
		util::log("CudaBlockSparseCache") << " slow allocate of "
			<< address << " with " << bytes << " bytes\n";
		
		return address;
	}
	
	void slowDeviceFree(void* address)
	{
		util::log("CudaBlockSparseCache") << " slow free of " << address << "\n";
		CudaDriver::cuMemFree((CUdeviceptr)address);
	}

	PoolAllocation* tryFastAllocation(size_t bytes)
	{
		// fast fail
		if(bytes > memoryPoolSize)
		{
			util::log("CudaBlockSparseCache") << " failed, pool isn't big enough\n";
			
			return nullptr;
		}
		
		// check the last entry
		if(!poolAllocations.empty())
		{
			auto lastEntry = poolAllocations.rbegin();
			
			size_t endOffset = align(getOffset(lastEntry->second.address) + lastEntry->second.size);
			size_t remaining = memoryPoolSize - endOffset;
			
			if(remaining >= bytes)
			{
				void* newAddress = (char*)memoryPoolBaseAddress + endOffset;
				auto newAllocation = poolAllocations.insert(
					std::make_pair(newAddress, PoolAllocation(newAddress, bytes))).first;
				
				return &newAllocation->second;
			}
			
			util::log("CudaBlockSparseCache") << " last entry (" << lastEntry->second.address
				<< ", " << lastEntry->second.size << ") doesn't have enough space.\n";
		}
		else
		{
			// trivial case
			void* newAddress = memoryPoolBaseAddress;
			auto newAllocation = poolAllocations.insert(
				std::make_pair(newAddress, PoolAllocation(newAddress, bytes))).first;
			
			return &newAllocation->second;
		}

		// TODO: faster than in order

		// scan in order
		for(auto entry = poolAllocations.begin(); entry != poolAllocations.end(); ++entry)
		{
			auto nextEntry = entry; ++nextEntry;
			if(nextEntry == poolAllocations.end()) break;
			
			size_t endOffset = align(getOffset(entry->second.address) + entry->second.size);
			size_t remaining = getOffset(nextEntry->second.address) - endOffset;
			
			if(remaining >= bytes)
			{
				void* newAddress = (char*)memoryPoolBaseAddress + endOffset;
				auto newAllocation = poolAllocations.insert(
					std::make_pair(newAddress, PoolAllocation(newAddress, bytes))).first;
				
				return &newAllocation->second;
			}
		}
		
		return nullptr;
	}
	
public:
	size_t getOffset(void* address)
	{
		return (size_t)address - (size_t)memoryPoolBaseAddress;
	}

	size_t align(size_t address)
	{
		const size_t alignment = 16;
		
		size_t remainder = address % alignment;
		return remainder == 0 ? address : address + alignment - remainder;
	}

public:
	bool isInitialized;
	
	void*  memoryPoolBaseAddress;
	size_t memoryPoolSize;
	
	PoolAllocationMap poolAllocations;

};

class Allocation
{
public:
	Allocation(void* a, size_t s)
	: address(a), size(s), dirty(false), hostDirty(false)
	{

	}

public:
	void* address;
	size_t size;

public:
	bool dirty;
	bool hostDirty;
};

typedef std::map<const BlockSparseMatrixImplementation*, Allocation> AllocationMap;

static size_t getMatrixSize(const BlockSparseMatrixImplementation* matrix)
{
	size_t size = matrix->size() * sizeof(float);
	
	// TODO: add buffer space for the last block
	
	return size;
}

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
		
		if(matrix->size() == 0) return nullptr;
		
		auto allocation = findExisting(matrix);
		
		if(allocation == nullptr)
		{
			allocation = createNewAllocation(matrix);
			
			cacheAllocation(matrix, allocation);
		}
		else
		{
			assert(allocation->size == getMatrixSize(matrix));
			
			if(allocation->hostDirty)
			{
				cacheAllocation(matrix, allocation);
				
				allocation->hostDirty = false;
			}
		}
	
		allocation->dirty = true;
	
		return reinterpret_cast<float*>(allocation->address);
	}

	float* acquireReadOnly(BlockSparseMatrixImplementation* matrix)
	{
		initialize();
		
		if(matrix->size() == 0) return nullptr;
		
		auto allocation = findExisting(matrix);
		
		if(allocation == nullptr)
		{
			allocation = createNewAllocation(matrix);
			
			cacheAllocation(matrix, allocation);
		}
		else
		{
			assert(allocation->size == getMatrixSize(matrix));
			
			if(allocation->hostDirty)
			{
				cacheAllocation(matrix, allocation);
				
				allocation->hostDirty = false;
			}
		}
	
		return reinterpret_cast<float*>(allocation->address);
	}

	float* acquireClobber(BlockSparseMatrixImplementation* matrix)
	{
		initialize();
		
		if(matrix->size() == 0) return nullptr;
		
		auto allocation = findExisting(matrix);
		
		if(allocation == nullptr)
		{
			allocation = createNewAllocation(matrix);
		}
		else
		{
			assert(allocation->size == getMatrixSize(matrix));
		}
		
		allocation->dirty = true;
		allocation->hostDirty = false;
		
		return reinterpret_cast<float*>(allocation->address);
	}
	
	void release(BlockSparseMatrixImplementation* matrix)
	{
		initialize();
		
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

			fastDeviceFree(allocation->second.address);
			
			allocations.erase(allocation);
		}
	}
	
	void synchronize(BlockSparseMatrixImplementation* matrix)
	{
		synchronizeHostReadOnly(matrix);
		
		auto allocation = allocations.find(matrix);
		
		if(allocation != allocations.end())
		{
			allocation->second.hostDirty = true;
		}
	}
	
	void synchronizeHostReadOnly(BlockSparseMatrixImplementation* matrix)
	{
		auto allocation = allocations.find(matrix);
		
		auto cudaMatrix = static_cast<CudaBlockSparseMatrix*>(matrix);
		
		if(allocation != allocations.end())
		{
			if(allocation->second.dirty)
			{	
				allocation->second.dirty = false;
			
				util::log("CudaBlockSparseCache") << "Flushing dirty data back to host for matrix "
					<< matrix << " (" << matrix->rows() << " rows, " << matrix->columns()
					<< " columns) on the GPU at (" << allocation->second.address << ", "
					<< allocation->second.size << " bytes).\n";
				
				// TODO: optimize this
				size_t position = 0;
				
				for(auto& block : cudaMatrix->rawData())
				{
					CudaDriver::cuMemcpyDtoH(block.data().data(),
						(CUdeviceptr)(position + (uint8_t*)allocation->second.address),
						sizeof(float) * block.size());
					
					position += sizeof(float) * block.size();
				}
			}
		}
	}

public:
	void* fastDeviceAllocate(size_t bytes)
	{
		return _memoryManager.fastDeviceAllocate(bytes);
	}

	void fastDeviceFree(void* address)
	{
		_memoryManager.fastDeviceFree(address);
	}

private:
	Allocation* createNewAllocation(BlockSparseMatrixImplementation* matrix)
	{
		size_t size = getMatrixSize(matrix);

		void* newMemory = nullptr;

		if(size > 0)
		{
			newMemory = fastDeviceAllocate(size);
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
		
		_memoryManager.initialize(maximumSize);
	}

private:
	DeviceMemoryManager _memoryManager;

};

static std::unique_ptr<CacheManager> cacheManager(new CacheManager);

float* CudaBlockSparseCache::acquire(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->acquire(matrix);
}

float* CudaBlockSparseCache::acquireReadOnly(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->acquireReadOnly(matrix);
}

float* CudaBlockSparseCache::acquireClobber(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->acquireClobber(matrix);
}

void CudaBlockSparseCache::release(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->release(matrix);
}

void CudaBlockSparseCache::invalidate(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->invalidate(matrix);
}

void CudaBlockSparseCache::synchronize(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->synchronize(matrix);
}

void CudaBlockSparseCache::synchronizeHostReadOnly(const BlockSparseMatrixImplementation* m)
{
	auto matrix = const_cast<BlockSparseMatrixImplementation*>(m);
	
	return cacheManager->synchronizeHostReadOnly(matrix);
}

void* CudaBlockSparseCache::fastDeviceAllocate(size_t bytes)
{
	return cacheManager->fastDeviceAllocate(bytes);
}

void CudaBlockSparseCache::fastDeviceFree(void* address)
{
	cacheManager->fastDeviceFree(address);
}

}

}


