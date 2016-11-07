
// Lucius Includes
#include <lucius/matrix/interface/SortOperations.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/Dimension.h>
#include <lucius/matrix/interface/DimensionTransformations.h>

#include <lucius/util/interface/debug.h>

namespace lucius
{
namespace matrix
{
namespace detail
{

CUDA_DECORATOR bool compare(const PackedKeysAndIndices& right,
    const Comparator& comparator) const
{
    if(comparator(dimensionKey, right.dimensionKey))
    {
        return true;
    }
    else if(comparator(right.dimensionKey, dimensionKey))
    {
        return false;
    }

    return comparator(normalKey, right.normalKey);
}

template <typename NativeType>
class PackedKeysAndIndices
{
public:
    PackedKeysAndIndices(NativeType dimensionKey, NativeType normalKey, size_t position);

public:
    NativeType dimensionKey;
    NativeType normalKey;

    size_t position;

};

template <typename NativeType>
class GatherKeysAndIndicesLambda
{
public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            auto position = linearToDimension(element, keysView.size());

            packedKeysAndIndices[element] = PackedKeysAndIndices(keysView(position),
                dimensionKeysView(position), element);
        }
    }

public:
    PackedKeysAndIndices* packedKeysAndIndices;

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
    auto gatherLambda = GatherKeysAndIndicesLambda<NativeType>{
        reinterpret_cast<SortElement*>(keysAndIndices.data()),
        keysView, dimensionKeysView, elements};

    parallel::multiBulkSynchronousParallel(gatherLambda);
}

template <typename OperationType, typename NativeType>
class BlockSortLambda
{
public:
    CUDA_DECORATOR void operator()(const parallel::TheadGroup& threadGroup) const
    {
        auto innerGroup    = parallel::partitionThreadGroupAtLevel(threadGroup, 2);
        auto relativeGroup = parallel::getRelativeGroup(innerGroup, threadGroup);

        for(size_t dataBlock = relativeGroup.id(); dataBlock < totalDataBlocks;
            dataBlock += relativeGroup.size())
        {
            LocalStorage localStorage;

            loadLocalStorage(localStorage, dataBlock, innerGroup);

            sortLocalStorage(localStorage);

            auto* memory = parallel::SharedMemoryAllocator<NativeType,
                LocalValueCount * parallel::GroupLevelSize<2>::size()>::allocate();

            copyFromLocalStorageIntoSharedStorage(memory, localStorage, innerGroup);

            mergeSortShared(memory, innerGroup);

            saveShared(memory, threadGroup);
        }
    }

    CUDA_DECORATOR void loadLocalStorage(LocalStorage& localStorage,
        size_t dataBlock, const parallel::ThreadGroup& threadGroup) const
    {


    }

public:
    typedef NativeType[Config::ValuesPerThread] LocalStorage;

public:
    typedef PackedKeysAndIndices<NativeType> SortElement;

public:
    SortElement* data;

public:
    OperationType nativeOperation;
};

template <typename OperationType, typename NativeType>
void blockSort(Allocation& keysAndIndices, const OperationType& operation)
{
    typedef PackedKeysAndIndices<PrecisionType, OperationType> SortElement;

    auto blockSortLambda = BlockSortLambda<OperationType, NativeType>{
        reinterpret_cast<SortElement*>(keysAndIndices.data()),
        operation};

    parallel::multiBulkSynchronousParallel(blockSortLambda);
}

template <typename OperationType, typename NativeType>
class MergeLambda
{
public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& group) const
    {
        // TODO
    }

public:
    typedef PackedKeysAndIndices<NativeType> SortElement;

public:
    SortElement*       keysAndIndicesOutput;
    const SortElement* keysAndIndicesInput;

public:
    OperationType operation;
};

template <typename OperationType, typename NativeType>
void merge(Allocation& keysAndIndicesOutput, const Allocation& keysAndIndicesInput,
    size_t blockTileSize, size_t elements, const OperationType& operation)
{
    typedef PackedKeysAndIndices<NativeType> SortElement;

    auto mergeLambda = MergeLambda<OperationType, NativeType>{
        reinterpret_cast<SortElement*>(keysAndIndicesOutput.data()),
        reinterpret_cast<const SortElement*>(keysAndIndicesInput.data()),
        operation};

    parallel::multiBulkSynchronousParallel(mergeLambda);
}

template <typename NativeType>
class GatherResultsLambda
{
public:
    CUDA_DECORATOR void operator()(const parallel::ThreadGroup& threadGroup) const
    {
        for(size_t element = threadGroup.id(); element < elements; element += threadGroup.size())
        {
            auto outputPosition = linearToDimension(element, keysOutputView.size());

            size_t inputElement = keysAndIndices.position;

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

template <typepname NativeType>
void gatherResults(MatrixView<NativeType>& keysOutputView,
    MatrixView<NativeType&>& valuesOutputView,
    ConstMatrixView<NativeType>& keysInputView, ConstMatrixView<NativeType&>& valuesInputView,
    const Allocation& keysAndIndices, size_t elements)
{
    typedef PackedKeysAndIndices<NativeType> SortElement;

    auto gatherResultsLambda = GatherResultsLambda<NativeType>{keysOutputView, valuesOutputView,
        keysInputView, valuesInputView,
        reinterpret_cast<const SortElement*>(keysAndIndices.data()), elements};

    parallel::multiBulkSynchronousParallel(gatherResultsLambda);
}

template <typename OperationType, typename PrecisionType>
void sortByKey(Matrix& keys, Matrix& values, const Matrix& dimensionKeys,
    const Operation& operation, const std::tuple<PrecisionType>& precisions)
{
    assert(PrecisionType() == keys.precision());
    assert(PrecisionType() == values.precision());
    assert(PrecisionType() == dimensionKeys.precision());

    typedef typename PrecisionType::type NativeType;

    auto& nativeOperation = static_cast<const OperationType&>(op);

    typedef PackedKeysAndIndices<NativeType> SortElement;

    size_t elements = keys.size().product();

    Allocation keysAndIndices(sizeof(SortElement) * elements);
    Allocation keysAndIndicesWorkspace(keysAndIndices.size());

    MatrixView<NativeType>      keysView(keys);
    MatrixView<NativeType>      valuesView(values);
    ConstMatrixView<NativeType> dimensionKeysView(dimensionKeys);

    gatherKeysAndIndices<NativeType>(keysAndIndices, keysView, dimensionKeysView, elements);

    blockSort<OperationType, NativeType>(keysAndIndices, nativeOperation);

    size_t blockSortTileSize = getBlockSortTileSize<PrecisionType>(elements);

    for(; blockSortTileSize < elements; blockSortTileSize *= 2)
    {
        merge<OperationType, NativeType>(keysAndIndicesWorkspace, keysAndIndices,
            blockTileSize, elements, nativeOperation);

        std::swap(keysAndIndices, keysAndIndicesWorkspace);
    }

    Matrix inputKeys   = copy(keys);
    Matrix inputValues = copy(values);

    ConstMatrixView<NativeType> keysInputView(inputKeys);
    ConstMatrixView<NativeType> valuesInputView(inputValues);

    gatherResults<NativeType>(keysView, valuesView, keysInputView, valuesInputView,
        keysAndIndices, elements);
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

    if(op == PossibleOperationType())
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

    Matrix& dimensionKeys(keys.size(), keys.precision());

    auto dimensionOrder = range(remainingDimensions, values.precision());
    broadcast(dimensionKeys, dimensionKeys, dimensionOrder, CopyRight());

    detail::sortByKey(keys, values, dimensionKeys, operation, AllComparisons());
}

void sortByKey(Matrix& keys, Matrix& values, const Operation& operation)
{
    sortByKey(keys, values, range(values.size()), operation);
}

}
}


