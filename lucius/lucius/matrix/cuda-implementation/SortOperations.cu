
// Lucius Includes
#include <lucius/matrix/interface/SortOperations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Allocation.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/Operation.h>
#include <lucius/matrix/interface/CopyOperations.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/DimensionTransformations.h>

#include <lucius/parallel/interface/MultiBulkSynchronousParallel.h>
#include <lucius/parallel/interface/SharedMemoryAllocator.h>
#include <lucius/parallel/interface/ScalarOperations.h>
#include <lucius/parallel/interface/Debug.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <limits>

namespace lucius
{
namespace matrix
{
namespace detail
{

class SortConfiguration
{
public:
    static constexpr size_t LocalValueCount = 4;
    static constexpr size_t GroupLevel      = 2;
};

template <typename NativeType>
class PackedKeysAndIndices
{
public:
    CUDA_DECORATOR PackedKeysAndIndices()
    {
    }

    CUDA_DECORATOR PackedKeysAndIndices(NativeType dimensionKey, NativeType normalKey,
        size_t position)
    : dimensionKey(dimensionKey), normalKey(normalKey), position(position)
    {

    }

public:
    NativeType dimensionKey;
    NativeType normalKey;

    size_t position;

};

template <typename NativeType, typename Comparator>
CUDA_DECORATOR bool compare(const PackedKeysAndIndices<NativeType>& left,
    const PackedKeysAndIndices<NativeType>& right,
    const Comparator& comparator)
{
    if(left.dimensionKey < right.dimensionKey)
    {
        return true;
    }
    else if(right.dimensionKey < left.dimensionKey)
    {
        return false;
    }

    return comparator(left.normalKey, right.normalKey);
}

template <typename NativeType>
class GatherKeysAndIndicesLambda
{
public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            auto position = linearToDimension(element, keysView.size());

            packedKeysAndIndices[element] = SortElement(dimensionKeysView(position),
                keysView(position), element);
        }
    }

public:
    typedef PackedKeysAndIndices<NativeType> SortElement;

    SortElement* packedKeysAndIndices;

public:
    MatrixView<NativeType>      keysView;
    ConstMatrixView<NativeType> dimensionKeysView;

public:
    size_t elements;

};

template <typename NativeType>
void gatherKeysAndIndices(Allocation& keysAndIndices, const MatrixView<NativeType>& keysView,
    const ConstMatrixView<NativeType>& dimensionKeysView, size_t elements)
{
    typedef PackedKeysAndIndices<NativeType> SortElement;

    auto gatherLambda = GatherKeysAndIndicesLambda<NativeType>{
        reinterpret_cast<SortElement*>(keysAndIndices.data()),
        keysView, dimensionKeysView, elements};

    parallel::multiBulkSynchronousParallel(gatherLambda);
}

template <typename OperationType, typename NativeType>
class BlockSortLambda
{
public:
    typedef PackedKeysAndIndices<NativeType> SortElement;

public:
    static constexpr size_t LocalValueCount = SortConfiguration::LocalValueCount;
    static constexpr size_t GroupLevel = SortConfiguration::GroupLevel;

public:
    static constexpr NativeType NativeMax = std::numeric_limits<NativeType>::max();
    static constexpr size_t     SizeMax   = std::numeric_limits<size_t>::max();

public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        auto innerGroup    = parallel::partitionThreadGroupAtLevel(threadGroup, GroupLevel);
        auto relativeGroup = parallel::getRelativeGroup(innerGroup, threadGroup);

        size_t dataBlockSize   = innerGroup.size() * LocalValueCount;
        size_t totalDataBlocks = (elements + dataBlockSize - 1) / dataBlockSize;

        for(size_t dataBlock = relativeGroup.id(); dataBlock < totalDataBlocks;
            dataBlock += relativeGroup.size())
        {
            SortElement localStorage[LocalValueCount];

            loadLocalStorage(localStorage, dataBlock, innerGroup);

            sortLocalStorage(localStorage);

            constexpr int sharedSize = LocalValueCount * parallel::GroupLevelSize<2>::size();
            CUDA_SHARED_DECORATOR SortElement memory[sharedSize];
                //parallel::SharedMemoryAllocator<SortElement,
                //LocalValueCount * parallel::GroupLevelSize<2>::size()>().allocate();

            copyFromLocalStorageIntoSharedStorage(memory, localStorage, innerGroup);

            barrier(innerGroup);

            mergeSortShared(memory, innerGroup);

            saveShared(memory, dataBlock, innerGroup);
        }
    }

public:
    CUDA_DECORATOR void loadLocalStorage(SortElement* localStorage,
        size_t dataBlock, const parallel::ThreadGroup& threadGroup) const
    {
        size_t dataBlockSize = LocalValueCount * threadGroup.size();
        size_t startPosition = dataBlock * dataBlockSize + threadGroup.id() * LocalValueCount;

        SortElement defaultValue(NativeMax, NativeMax, SizeMax);

        for(size_t value = 0; value < LocalValueCount; ++value)
        {
            if(!_isInRange(startPosition + value))
            {
                localStorage[value] = defaultValue;
            }
            else
            {
                localStorage[value] = data[startPosition + value];
            }
        }
    }

    CUDA_DECORATOR bool _isInRange(size_t position) const
    {
        return position < elements;
    }

    CUDA_DECORATOR void sortLocalStorage(SortElement* localStorage) const
    {
        // Even odd transposition sort (for stability)
        for(size_t level = 0; level < LocalValueCount; ++level)
        {
            bool isOdd = level % 2 == 1;

            if(isOdd)
            {
                size_t range = (LocalValueCount - 1) / 2;

                for(size_t index = 0; index < range; ++index)
                {
                    size_t i = index * 2 + 1;

                    if(compare(localStorage[i+1], localStorage[i], comparisonOperation))
                    {
                        parallel::swap(localStorage[i+1], localStorage[i]);
                    }
                }
            }
            else
            {
                size_t range = (LocalValueCount) / 2;

                for(size_t index = 0; index < range; ++index)
                {
                    size_t i = index * 2;

                    if(compare(localStorage[i+1], localStorage[i], comparisonOperation))
                    {
                        parallel::swap(localStorage[i+1], localStorage[i]);
                    }
                }
            }
        }
    }

public:
    CUDA_DECORATOR void mergeSortShared(SortElement* sharedMemory,
        const parallel::ThreadGroup& innerGroup) const
    {
        for(size_t phase = 1, phaseSize = 2;
            phaseSize < innerGroup.size(); ++phase, phaseSize *= 2)
        {
            mergeRegions(sharedMemory, phase, innerGroup);
        }
    }

    CUDA_DECORATOR void mergeRegions(SortElement* sharedMemory,
        size_t phase, const parallel::ThreadGroup& innerGroup) const
    {
        SortElement* aBegin = nullptr;
        SortElement* aEnd   = nullptr;

        SortElement* bBegin = nullptr;
        SortElement* bEnd   = nullptr;

        size_t threadsPerResult = power(2, phase);

        auto localGroup = partitionThreadGroup(innerGroup, threadsPerResult);
        auto relativeGroup = getRelativeGroup(localGroup, innerGroup);

        size_t segmentSize = LocalValueCount * localGroup.size();

        size_t idInSegment     = localGroup.id();
        size_t halfSegmentSize = segmentSize / 2;
        size_t halfSegmentId   = 2 * relativeGroup.id();

        SortElement* a = sharedMemory +  halfSegmentId      * halfSegmentSize;
        SortElement* b = sharedMemory + (halfSegmentId + 1) * halfSegmentSize;

        // Merge path
        mergePath(aBegin, aEnd, bBegin, bEnd, a, b, idInSegment, halfSegmentSize);

        // Serial merge
        SortElement outputs[LocalValueCount];

        if((aEnd - aBegin) > 0 && (bEnd - bBegin) > 0)
        {
            #if !defined(LUCIUS_DEBUG)
            parallel::log("SortOperations") << "thread " << innerGroup.id() << " serial merge a["
                << (aBegin - a) << ", " << (aEnd - a) << "], b [" << (bBegin - b) << ", "
                << (bEnd - b) << "] half segment id " << halfSegmentId
                << " with size " << halfSegmentSize << "\n";
            #endif
        }

        serialMerge(outputs, aBegin, aEnd, bBegin, bEnd, innerGroup);

        barrier(innerGroup);

        // write results back to shared
        copyFromLocalStorageIntoSharedStorage(sharedMemory, outputs, innerGroup);

        barrier(innerGroup);
    }

public:
    CUDA_DECORATOR size_t power(size_t base, size_t exponent) const
    {
        size_t result = 1;

        while(exponent > 0)
        {
            result *= base;
            exponent -= 1;
        }

        return result;
    }

    CUDA_DECORATOR void mergePath(SortElement*& aBegin, SortElement*& aEnd,
        SortElement*& bBegin, SortElement*& bEnd, SortElement* a,
        SortElement* b, size_t segment, size_t segmentSize) const
    {
        size_t leftDiagonalConstraint  = (segment)     * LocalValueCount;
        size_t rightDiagonalConstraint = (segment + 1) * LocalValueCount;

        size_t left  = mergePath(a, b, leftDiagonalConstraint,  segmentSize);
        size_t right = mergePath(a, b, rightDiagonalConstraint, segmentSize);

        aBegin = a + left;
        aEnd   = a + right;

        bBegin = b + leftDiagonalConstraint - left;
        bEnd   = b + rightDiagonalConstraint - right;
    }

    CUDA_DECORATOR size_t mergePath(SortElement* a,
        SortElement* b, size_t diagonalConstraint, size_t segmentSize) const
    {
        size_t begin = diagonalConstraint > segmentSize ? diagonalConstraint - segmentSize : 0;
        size_t end   = parallel::min(diagonalConstraint, segmentSize);

        while(begin < end)
        {
            size_t midPoint = (begin + end) / 2;

            auto aKey = a[midPoint];
            auto bKey = b[diagonalConstraint - 1 - midPoint];

            bool predicate = compare(aKey, bKey, comparisonOperation);

            if(predicate)
            {
                begin = midPoint + 1;
            }
            else
            {
                end = midPoint;
            }
        }

        return begin;
    }

    CUDA_DECORATOR void serialMerge(SortElement* outputs, SortElement* aBegin, SortElement* aEnd,
        SortElement* bBegin, SortElement* bEnd, const parallel::ThreadGroup& innerGroup) const
    {
        size_t aIndex = 0;
        size_t bIndex = 0;

        size_t aLength = aEnd - aBegin;
        size_t bLength = bEnd - bBegin;

        for(size_t index = 0; index < LocalValueCount; ++index)
        {
            bool isALegal = aIndex < aLength;
            bool isBLegal = bIndex < bLength;

            bool isA = isALegal;

            if(isALegal && isBLegal)
            {
                isA = compare(aBegin[aIndex], bBegin[bIndex], comparisonOperation);
            }

            if(isA)
            {
                outputs[index] = aBegin[aIndex++];
            }
            else if(isBLegal)
            {
                outputs[index] = bBegin[bIndex++];
            }
        }
    }

public:
    CUDA_DECORATOR void copyFromLocalStorageIntoSharedStorage(SortElement* sharedMemory,
        const SortElement* localStorage, const parallel::ThreadGroup& innerGroup) const
    {
        size_t base = LocalValueCount * innerGroup.id();

        for(size_t index = 0; index < LocalValueCount; ++index)
        {
            auto value = localStorage[index];

            sharedMemory[base + index] = value;
        }
    }

    CUDA_DECORATOR void saveShared(SortElement* sharedMemory, size_t blockId,
        const parallel::ThreadGroup& innerGroup) const
    {
        size_t sharedElement = innerGroup.id();
        size_t globalElement = blockId * LocalValueCount * innerGroup.size() + innerGroup.id();

        for(size_t element = 0; element < LocalValueCount; ++element)
        {
            if(globalElement < elements)
            {
                auto value = sharedMemory[sharedElement];

                #if !defined(LUCIUS_DEBUG)
                parallel::log("SortOperations") << "block sort output[" << globalElement
                    << "] = (" << value.dimensionKey << ", " << value.normalKey << ")\n";
                #endif

                data[globalElement] = value;
            }

            globalElement += innerGroup.size();
            sharedElement += innerGroup.size();
        }
    }

public:
    typedef NativeType LocalStorage[LocalValueCount];

public:
    SortElement* data;

public:
    size_t elements;

public:
    OperationType comparisonOperation;
};

template <typename OperationType, typename NativeType>
void blockSort(Allocation& keysAndIndices, size_t elements, const OperationType& operation)
{
    typedef PackedKeysAndIndices<NativeType> SortElement;

    auto blockSortLambda = BlockSortLambda<OperationType, NativeType>{
        reinterpret_cast<SortElement*>(keysAndIndices.data()),
        elements,
        operation};

    parallel::multiBulkSynchronousParallel(blockSortLambda);
}

template <typename OperationType, typename NativeType>
class MergeLambda
{
public:
    typedef PackedKeysAndIndices<NativeType> SortElement;

public:
    static constexpr size_t LocalValueCount = SortConfiguration::LocalValueCount;
    static constexpr size_t GroupLevel      = SortConfiguration::GroupLevel;

public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        auto innerGroup    = parallel::partitionThreadGroupAtLevel(threadGroup, GroupLevel);
        auto relativeGroup = parallel::getRelativeGroup(innerGroup, threadGroup);

        size_t elementsPerInnerGroup = innerGroup.size() * LocalValueCount;

        size_t mergedRegionSize = regionToMergeSize * 2;
        size_t totalMergeGroups = (elements + mergedRegionSize - 1) / mergedRegionSize;
        size_t totalRegionsPerMergeGroup = (mergedRegionSize + elementsPerInnerGroup - 1) /
            elementsPerInnerGroup;

        size_t innerGroupsPerMergeGroup = parallel::min(totalRegionsPerMergeGroup,
            relativeGroup.size());
        size_t mergeGroupSize = innerGroupsPerMergeGroup * innerGroup.size();

        auto mergeGroup = parallel::partitionThreadGroup(threadGroup, mergeGroupSize);
        auto relativeInnerToMergeGroup = parallel::getRelativeGroup(innerGroup, mergeGroup);
        auto relativeMergeToOuterGroup = parallel::getRelativeGroup(mergeGroup, threadGroup);

        constexpr int sharedSize = LocalValueCount * parallel::GroupLevelSize<2>::size();
        CUDA_SHARED_DECORATOR SortElement sharedMemory[sharedSize];
        //auto* sharedMemory = parallel::SharedMemoryAllocator<SortElement,
        //    LocalValueCount * parallel::GroupLevelSize<2>::size()>().allocate();

        SortElement localStorage[LocalValueCount];

        for(size_t mergeGroupId = relativeMergeToOuterGroup.id();
            mergeGroupId < totalMergeGroups;
            mergeGroupId += relativeMergeToOuterGroup.size())
        {
            size_t mergeGroupABegin = mergeGroupId * mergedRegionSize;
            size_t mergeGroupAEnd   = parallel::min(mergeGroupABegin + regionToMergeSize,
                                                    elements);

            size_t mergeGroupBBegin = mergeGroupAEnd;
            size_t mergeGroupBEnd   = parallel::min(mergeGroupBBegin + regionToMergeSize,
                                                    elements);

            size_t remainingElements = elements - mergeGroupABegin;
            size_t regionsPerMergeGroup = totalRegionsPerMergeGroup;

            if(remainingElements < mergedRegionSize)
            {
                regionsPerMergeGroup = (remainingElements + elementsPerInnerGroup - 1) /
                    elementsPerInnerGroup;
            }

            for(size_t innerGroupId = relativeInnerToMergeGroup.id();
                innerGroupId < regionsPerMergeGroup;
                innerGroupId += relativeInnerToMergeGroup.size())
            {
                // group merge path
                size_t innerGroupLeftDiagonal  =
                    parallel::min( innerGroupId * elementsPerInnerGroup, remainingElements);
                size_t innerGroupRightDiagonal =
                    parallel::min((innerGroupId + 1) * elementsPerInnerGroup, remainingElements);

                const SortElement* aBegin = nullptr;
                const SortElement* aEnd   = nullptr;

                _mergePathForEntireGroup(aBegin, aEnd, mergeGroupABegin,
                    mergeGroupAEnd, mergeGroupBBegin, mergeGroupBEnd, innerGroupLeftDiagonal,
                    innerGroupRightDiagonal, innerGroup);

                _copyIntoSharedMemory(sharedMemory, aBegin, aEnd,
                    mergeGroupABegin, mergeGroupBBegin,
                    innerGroupLeftDiagonal, innerGroupRightDiagonal, innerGroup);

                // thread merge path
                size_t aSize = aEnd - aBegin;

                size_t aBeginOffset = aBegin - keysAndIndicesInput - mergeGroupABegin;
                size_t aEndOffset   = aEnd   - keysAndIndicesInput - mergeGroupABegin;

                size_t bBeginOffset = innerGroupLeftDiagonal  - aBeginOffset;
                size_t bEndOffset   = innerGroupRightDiagonal - aEndOffset;

                size_t bSize = bEndOffset - bBeginOffset;

                SortElement* sharedABegin = nullptr;
                SortElement* sharedAEnd   = nullptr;

                size_t remainingThreadElements = remainingElements -
                    innerGroupId * elementsPerInnerGroup;

                size_t threadLeftDiagonal  = parallel::min( innerGroup.id()      * LocalValueCount,
                    remainingThreadElements);
                size_t threadRightDiagonal = parallel::min((innerGroup.id() + 1) * LocalValueCount,
                    remainingThreadElements);

                _mergePathPerThread(sharedABegin, sharedAEnd, aEnd - aBegin, sharedMemory,
                    bSize + aSize, threadLeftDiagonal, threadRightDiagonal, innerGroup);

                size_t threadABeginOffset = sharedABegin - sharedMemory;
                size_t threadAEndOffset   = sharedAEnd   - sharedMemory;

                size_t threadBBeginOffset =  threadLeftDiagonal - threadABeginOffset;
                size_t threadBEndOffset   = threadRightDiagonal - threadAEndOffset;

                #if !defined(LUCIUS_DEBUG)
                parallel::log("SortOperations") << "per block merge path a("
                    << aBeginOffset << ", " << aEndOffset << "), b("
                    << bBeginOffset << ", " << bEndOffset << "), per thread a("
                    << threadABeginOffset << ", " << threadAEndOffset << "), b("
                    << threadBBeginOffset << ", " << threadBEndOffset << ")\n";
                #endif

                _serialMerge(localStorage, sharedABegin, sharedAEnd, sharedMemory,
                    sharedMemory + aSize, threadLeftDiagonal, threadRightDiagonal, innerGroup);

                size_t outputStart = mergeGroupId * mergedRegionSize +
                    innerGroupId * elementsPerInnerGroup +
                    innerGroup.id() * LocalValueCount;

                _saveOutputs(localStorage, outputStart, innerGroup);
            }
        }
    }

private:
    CUDA_DECORATOR void _mergePathForEntireGroup(const SortElement*& aBegin,
        const SortElement*& aEnd,
        size_t mergeGroupABegin, size_t mergeGroupAEnd, size_t mergeGroupBBegin,
        size_t mergeGroupBEnd, size_t leftDiagonal, size_t rightDiagonal,
        const parallel::ThreadGroup& group) const
    {
        size_t aBeginOffset = _mergePathForEntireGroup(mergeGroupABegin, mergeGroupAEnd,
            mergeGroupBBegin, mergeGroupBEnd, leftDiagonal, group);
        size_t aEndOffset   = _mergePathForEntireGroup(mergeGroupABegin, mergeGroupAEnd,
            mergeGroupBBegin, mergeGroupBEnd, rightDiagonal, group);

        aBegin = keysAndIndicesInput + mergeGroupABegin + aBeginOffset;
        aEnd   = keysAndIndicesInput + mergeGroupABegin +   aEndOffset;
    }

    CUDA_DECORATOR size_t _mergePathForEntireGroup(
        size_t mergeGroupABegin, size_t mergeGroupAEnd, size_t mergeGroupBBegin,
        size_t mergeGroupBEnd, size_t diagonal, const parallel::ThreadGroup& group) const
    {
        const SortElement* a = keysAndIndicesInput + mergeGroupABegin;
        const SortElement* b = keysAndIndicesInput + mergeGroupBBegin;

        size_t aSize = mergeGroupAEnd - mergeGroupABegin;
        size_t bSize = mergeGroupBEnd - mergeGroupBBegin;

        size_t begin = diagonal > bSize ? diagonal - bSize : 0;
        size_t end   = parallel::min(diagonal, aSize);

        while(begin < end)
        {
            size_t midPoint = (begin + end) / 2;

            SortElement aKey = a[midPoint];
            SortElement bKey = b[diagonal - 1 - midPoint];

            bool predicate = compare(aKey, bKey, comparisonOperation);

            if(predicate)
            {
                begin = midPoint + 1;
            }
            else
            {
                end = midPoint;
            }
        }

        return begin;
    }

private:
    CUDA_DECORATOR void _copyIntoSharedMemory(SortElement* sharedMemory, const SortElement* aBegin,
        const SortElement* aEnd, size_t mergeGroupABegin, size_t mergeGroupBBegin,
        size_t leftDiagonal, size_t rightDiagonal, const parallel::ThreadGroup& innerGroup) const
    {
        size_t totalASize = aEnd - aBegin;

        SortElement* sharedMemoryB = sharedMemory + totalASize;

        const SortElement* bBase = keysAndIndicesInput + mergeGroupBBegin;

        size_t aStartOffset = aBegin - keysAndIndicesInput - mergeGroupABegin;
        size_t aEndOffset   = aStartOffset + totalASize;

        auto* bBegin = bBase + leftDiagonal   - aStartOffset;
        auto* bEnd   = bBase + rightDiagonal  - aEndOffset;

        _copyIntoSharedMemory(sharedMemory,  aBegin, aEnd, innerGroup);
        _copyIntoSharedMemory(sharedMemoryB, bBegin, bEnd, innerGroup);
    }

    CUDA_DECORATOR void _copyIntoSharedMemory(SortElement* sharedMemory, const SortElement* begin,
        const SortElement* end, const parallel::ThreadGroup& innerGroup) const
    {
        size_t size = end - begin;

        for(size_t index = innerGroup.id(); index < size; index += innerGroup.size())
        {
            sharedMemory[index] = begin[index];
        }
    }

private:
    CUDA_DECORATOR void _mergePathPerThread(
        SortElement*& sharedABegin, SortElement*& sharedAEnd,
        size_t aSize, SortElement* sharedMemory, size_t sharedMemorySize,
        size_t leftDiagonal, size_t rightDiagonal, const parallel::ThreadGroup& group) const
    {
        auto* aBegin = sharedMemory;
        auto* aEnd   = aBegin + aSize;

        auto* bBegin = aEnd;
        auto* bEnd   = sharedMemory + sharedMemorySize;

        size_t aBeginOffset = _mergePathPerThread(aBegin, aEnd, bBegin, bEnd,  leftDiagonal);
        size_t aEndOffset   = _mergePathPerThread(aBegin, aEnd, bBegin, bEnd, rightDiagonal);

        sharedABegin = aBegin + aBeginOffset;
        sharedAEnd   = aBegin + aEndOffset;
    }

    CUDA_DECORATOR size_t _mergePathPerThread(const SortElement* aBegin, const SortElement* aEnd,
        const SortElement* bBegin, const SortElement* bEnd, size_t diagonal) const
    {
        const SortElement* a = aBegin;
        const SortElement* b = bBegin;

        size_t aSize = aEnd - aBegin;
        size_t bSize = bEnd - bBegin;

        size_t begin = diagonal > bSize ? diagonal - bSize : 0;
        size_t end   = parallel::min(diagonal, aSize);

        while(begin < end)
        {
            size_t midPoint = (begin + end) / 2;

            SortElement aKey = a[midPoint];
            SortElement bKey = b[diagonal - 1 - midPoint];

            bool predicate = compare(aKey, bKey, comparisonOperation);

            if(predicate)
            {
                begin = midPoint + 1;
            }
            else
            {
                end = midPoint;
            }
        }

        return begin;
    }

private:
    CUDA_DECORATOR void _serialMerge(SortElement* localStorage,
        const SortElement* sharedABegin, const SortElement* sharedAEnd,
        const SortElement* sharedBase, const SortElement* bBase, size_t leftDiagonal,
        size_t rightDiagonal, const parallel::ThreadGroup& innerGroup) const
    {
        size_t sharedABeginOffset = sharedABegin - sharedBase;
        size_t sharedAEndOffset   = sharedAEnd   - sharedBase;

        size_t bBaseOffset = bBase - sharedBase;

        size_t sharedBBeginOffset = bBaseOffset + leftDiagonal  - sharedABeginOffset;
        size_t sharedBEndOffset   = bBaseOffset + rightDiagonal - sharedAEndOffset;

        const SortElement* a = sharedABegin;
        const SortElement* b = bBase;

        size_t aSize = sharedAEnd       - sharedABegin;
        size_t bSize = sharedBEndOffset - sharedBBeginOffset;

        size_t aOffset = 0;
        size_t bOffset = 0;

        for(size_t index = 0; index < LocalValueCount; ++index)
        {
            SortElement currentElement;

            if(aOffset < aSize)
            {
                if(bOffset < bSize && compare(b[bOffset], a[aOffset], comparisonOperation))
                {
                    currentElement = b[bOffset++];
                }
                else
                {
                    currentElement = a[aOffset++];
                }
            }
            else
            {
                currentElement = b[bOffset++];
            }

            localStorage[index] = currentElement;
        }
    }

private:
    CUDA_DECORATOR void _saveOutputs(const SortElement* localStorage, size_t outputStart,
        const parallel::ThreadGroup& innerGroup) const
    {
        for(size_t index = 0; index < LocalValueCount; ++index)
        {
            auto value = localStorage[index];

            if(outputStart + index < elements)
            {
                #if !defined(LUCIUS_DEBUG)
                parallel::log("SortOperations") << "merge output["
                    << (outputStart + index) << "] = ("
                    << value.dimensionKey << ", " << value.normalKey << ")\n";
                #endif

                keysAndIndicesOutput[outputStart + index] = value;
            }
        }
    }

public:
    SortElement*       keysAndIndicesOutput;
    const SortElement* keysAndIndicesInput;

public:
    size_t elements;
    size_t regionToMergeSize;

public:
    OperationType comparisonOperation;
};

template <typename OperationType, typename NativeType>
void merge(Allocation& keysAndIndicesOutput, const Allocation& keysAndIndicesInput,
    size_t blockTileSize, size_t elements, const OperationType& operation)
{
    typedef PackedKeysAndIndices<NativeType> SortElement;

    auto mergeLambda = MergeLambda<OperationType, NativeType>{
        reinterpret_cast<SortElement*>(keysAndIndicesOutput.data()),
        reinterpret_cast<const SortElement*>(keysAndIndicesInput.data()),
        elements, blockTileSize, operation};

    parallel::multiBulkSynchronousParallel(mergeLambda);
}

template <typename NativeType>
class GatherResultsLambda
{
public:
    typedef PackedKeysAndIndices<NativeType> SortElement;

public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            auto outputPosition = linearToDimension(element, keysOutputView.size());

            size_t inputElement = keysAndIndices[element].position;

            auto inputPosition = linearToDimension(inputElement, keysOutputView.size());

              keysOutputView(outputPosition) =   keysInputView(inputPosition);
            valuesOutputView(outputPosition) = valuesInputView(inputPosition);
        }
    }

public:
    MatrixView<NativeType> keysOutputView;
    MatrixView<NativeType> valuesOutputView;

public:
    ConstMatrixView<NativeType> keysInputView;
    ConstMatrixView<NativeType> valuesInputView;

public:
    const SortElement* keysAndIndices;

public:
    size_t elements;
};

template <typename NativeType>
void gatherResults(MatrixView<NativeType>& keysOutputView,
    MatrixView<NativeType>& valuesOutputView,
    ConstMatrixView<NativeType>& keysInputView, ConstMatrixView<NativeType>& valuesInputView,
    const Allocation& keysAndIndices, size_t elements)
{
    typedef PackedKeysAndIndices<NativeType> SortElement;

    auto gatherResultsLambda = GatherResultsLambda<NativeType>{keysOutputView, valuesOutputView,
        keysInputView, valuesInputView,
        reinterpret_cast<const SortElement*>(keysAndIndices.data()), elements};

    parallel::multiBulkSynchronousParallel(gatherResultsLambda);
}

template <typename NativeType>
size_t getBlockSortTileSize(size_t elements)
{
    size_t groupSize = parallel::GroupLevelSize<SortConfiguration::GroupLevel>::size();

    if(parallel::isCudaEnabled())
    {
        groupSize = parallel::GroupLevelSize<SortConfiguration::GroupLevel>::cudaSize();
    }

    return groupSize * SortConfiguration::LocalValueCount;
}

template <typename OperationType, typename PrecisionType>
void sortByKey(Matrix& keys, Matrix& values, const Matrix& dimensionKeys,
    const Operation& operation, const std::tuple<PrecisionType>& precisions)
{
    if(util::isLogEnabled("SortOperations::Detail"))
    {
        util::log("SortOperations::Detail") << " sorting keys: " <<
            keys.debugString();
        util::log("SortOperations::Detail") << " dimension keys: " <<
            dimensionKeys.debugString();
        util::log("SortOperations::Detail") << " sorting values: " <<
            values.debugString();
    }

    assert(PrecisionType() == keys.precision());
    assert(PrecisionType() == values.precision());
    assert(PrecisionType() == dimensionKeys.precision());

    assert(dimensionKeys.size() == keys.size());
    assert(keys.size() == values.size());

    typedef typename PrecisionType::type NativeType;

    auto& nativeOperation = static_cast<const OperationType&>(operation);

    typedef PackedKeysAndIndices<NativeType> SortElement;

    size_t elements = keys.size().product();

    Allocation keysAndIndices(sizeof(SortElement) * elements);
    Allocation keysAndIndicesWorkspace(keysAndIndices.size());

    MatrixView<NativeType>      keysView(keys);
    MatrixView<NativeType>      valuesView(values);
    ConstMatrixView<NativeType> dimensionKeysView(dimensionKeys);

    util::log("SortOperations::Detail") << " Gathering " << elements << " keys and indices.\n";

    gatherKeysAndIndices<NativeType>(keysAndIndices, keysView, dimensionKeysView, elements);

    size_t blockSortTileSize = getBlockSortTileSize<NativeType>(elements);

    util::log("SortOperations::Detail") << " Running block sort on groups of "
        << blockSortTileSize << " elements.\n";

    blockSort<OperationType, NativeType>(keysAndIndices, elements, nativeOperation);

    for(; blockSortTileSize < elements; blockSortTileSize *= 2)
    {
        util::log("SortOperations::Detail") << " Running tile merge on groups of "
            << blockSortTileSize << " elements.\n";
        merge<OperationType, NativeType>(keysAndIndicesWorkspace, keysAndIndices,
            blockSortTileSize, elements, nativeOperation);

        std::swap(keysAndIndices, keysAndIndicesWorkspace);
    }

    Matrix inputKeys   = copy(keys);
    Matrix inputValues = copy(values);

    ConstMatrixView<NativeType> keysInputView(inputKeys);
    ConstMatrixView<NativeType> valuesInputView(inputValues);

    gatherResults<NativeType>(keysView, valuesView, keysInputView, valuesInputView,
        keysAndIndices, elements);

    util::log("SortOperations::Detail") << " Gathering " << elements << " keys and indices.\n";

    if(util::isLogEnabled("SortOperations::Detail"))
    {
        util::log("SortOperations::Detail") << " sorted keys: " <<
            keys.debugString();
        util::log("SortOperations::Detail") << " sorted values: " <<
            values.debugString();
    }
}

template <typename OperationType, typename PossiblePrecisions>
void sortByKey(Matrix& keys, Matrix& values, const Matrix& dimensionKeys,
    const Operation& operation, const PossiblePrecisions& possiblePrecisions)
{
    typedef typename std::tuple_element<0, PossiblePrecisions>::type PossiblePrecisionType;

    if(PossiblePrecisionType() == keys.precision())
    {
        sortByKey<OperationType>(keys, values, dimensionKeys, operation,
            std::tuple<PossiblePrecisionType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossiblePrecisions>::type RemainingPrecisions;

        sortByKey<OperationType>(keys, values, dimensionKeys, operation,
            RemainingPrecisions());
    }

}

template <typename OperationType>
void sortByKey(Matrix& keys, Matrix& values, const Matrix& dimensionKeys,
    const Operation& operation, const std::tuple<OperationType>& operations)
{
    assert(OperationType() == operation);

    sortByKey<OperationType>(keys, values, dimensionKeys, operation, AllPrecisions());
}

template <typename PossibleOperations>
void sortByKey(Matrix& keys, Matrix& values, const Matrix& dimensionKeys,
    const Operation& operation, const PossibleOperations& possibleOperations)
{
    typedef typename std::tuple_element<0, PossibleOperations>::type PossibleOperationType;

    if(PossibleOperationType() == operation)
    {
        sortByKey(keys, values, dimensionKeys, operation, std::tuple<PossibleOperationType>());
    }
    else
    {
        typedef typename util::RemoveFirstType<PossibleOperations>::type RemainingOperations;

        sortByKey(keys, values, dimensionKeys, operation, RemainingOperations());
    }
}

} // namespace detail

void sort(Matrix& values, const Dimension& dimensionsToSort, const Operation& operation)
{
    sortByKey(values, values, dimensionsToSort, operation);
}

void sort(Matrix& values, const Operation& operation)
{
    sort(values, range(values.size()), operation);
}

void sortByKey(Matrix& keys, Matrix& values, const Dimension& dimensionsToSort,
    const Operation& operation)
{
    auto remainingDimensions = removeDimensions(values.size(), dimensionsToSort);

    Matrix dimensionKeys(keys.size(), keys.precision());

    auto dimensionOrder = range(remainingDimensions, values.precision());
    broadcast(dimensionKeys, dimensionKeys, dimensionOrder, dimensionsToSort, CopyRight());

    detail::sortByKey(keys, values, dimensionKeys, operation, AllComparisonOperations());
}

void sortByKey(Matrix& keys, Matrix& values, const Operation& operation)
{
    sortByKey(keys, values, range(values.size()), operation);
}

}
}


