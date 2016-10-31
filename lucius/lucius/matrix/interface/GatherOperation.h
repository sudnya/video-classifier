/*  \file   GatherOperation.h
    \date   October 30, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the GatherOperation classes.
*/

#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/ScalarOperations.h>

#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/MatrixView.h>

// Standard Library Includes
#include <cmath>
#include <algorithm>

namespace lucius
{
namespace matrix
{


/*! \brief A class for specifying matrix gather operations. */
class GatherOperation
{
public:
    enum Type
    {
        Pool2DGather,
        Pool2DGatherInverse,
        PermuteDimensionGather,
        HanningGather,
        GatherIndexToOneHot,
        MapOutputToIndexDimension,
        MapOutputToMatchingIndexDimension

    };

public:
    CUDA_DECORATOR GatherOperation(Type t) : _type(t) {}

public:
    CUDA_DECORATOR bool operator==(const GatherOperation&) const;

private:
    Type _type;

};

class SimpleGatherOperation : public GatherOperation
{
public:
    SimpleGatherOperation(Type t) : GatherOperation(t) {}
    virtual ~SimpleGatherOperation() {}

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperation(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView) const
    {
        size_t outputIndex = dotProduct(outputDimension, outputView.stride());

        size_t inputIndex = (*this)(outputIndex);

        return inputView(linearToDimension(inputIndex, inputView.size()));
    }

public:
    virtual size_t operator()(size_t) const { return 0; }

};

class Pool2DGather : public SimpleGatherOperation
{
public:
    CUDA_DECORATOR Pool2DGather()
    : SimpleGatherOperation(GatherOperation::Pool2DGather)
    {}

    CUDA_DECORATOR Pool2DGather(size_t w, size_t h, size_t inputW, size_t inputH)
    : SimpleGatherOperation(GatherOperation::Pool2DGather),
      width(w), height(h), inputWidth(inputW), inputHeight(inputH)
    {

    }

public:
    CUDA_DECORATOR size_t operator()(size_t outputPosition) const
    {
        size_t outputTilesW = inputWidth / width;

        size_t inputOffset = outputPosition % (inputWidth * inputHeight);
        size_t inputBase   = outputPosition - inputOffset;

        size_t outputTile = inputOffset / (width * height);
        size_t tileOffset = inputOffset % (width * height);

        size_t outputTileW = outputTile % outputTilesW;
        size_t outputTileH = outputTile / outputTilesW;

        size_t outputTileWOffset = tileOffset % width;
        size_t outputTileHOffset = tileOffset / width;

        size_t outputOffset = (outputTileW * width + outputTileWOffset) +
            (outputTileH * height + outputTileHOffset) * inputWidth;

        return outputOffset + inputBase;
    }

public:
    size_t width;
    size_t height;

    size_t inputWidth;
    size_t inputHeight;

};

class Pool2DGatherInverse : public SimpleGatherOperation
{
public:
    CUDA_DECORATOR Pool2DGatherInverse()
    : SimpleGatherOperation(GatherOperation::Pool2DGatherInverse)
    {}

    CUDA_DECORATOR Pool2DGatherInverse(size_t w, size_t h, size_t inputW, size_t inputH)
    : SimpleGatherOperation(GatherOperation::Pool2DGatherInverse),
      width(w), height(h), inputWidth(inputW), inputHeight(inputH)
    {

    }

public:
    CUDA_DECORATOR size_t operator()(size_t outputPosition) const
    {
        size_t inputOffset = outputPosition % (inputWidth * inputHeight);
        size_t inputBase   = outputPosition - inputOffset;

        size_t tileSize = width * height;

        size_t inputW = inputOffset % inputWidth;
        size_t inputH = inputOffset / inputWidth;

        size_t tileW = inputW / width;
        size_t tileWOffset = inputW % width;

        size_t tileH = inputH / height;
        size_t tileHOffset = inputH % height;

        size_t inputWidthInTiles = inputWidth / width;

        size_t tileId = tileW + tileH * inputWidthInTiles;

        size_t tileOffset = tileWOffset + tileHOffset * width;

        return inputBase + tileId * tileSize + tileOffset;
    }

public:
    size_t width;
    size_t height;

    size_t inputWidth;
    size_t inputHeight;

};

class PermuteDimensionGather : public SimpleGatherOperation
{
public:
    CUDA_DECORATOR PermuteDimensionGather()
    : SimpleGatherOperation(GatherOperation::PermuteDimensionGather)
    {}

    CUDA_DECORATOR PermuteDimensionGather(const Dimension& inputStride,
        const Dimension& outputSize, const Dimension& order)
    : SimpleGatherOperation(GatherOperation::PermuteDimensionGather), inputStride(inputStride),
      outputSize(outputSize), order(order)
    {

    }

public:
    CUDA_DECORATOR size_t operator()(size_t outputPosition) const
    {
        auto outputDimension = linearToDimension(outputPosition, outputSize);

        auto inputDimension = selectReverseMappingDimensions(outputDimension, order);

        return dotProduct(inputDimension, inputStride);
    }

public:
    Dimension inputStride;
    Dimension outputSize;
    Dimension order;

};

class HanningGather : public SimpleGatherOperation
{
public:
    CUDA_DECORATOR HanningGather()
    : SimpleGatherOperation(GatherOperation::HanningGather)
    {}

    CUDA_DECORATOR HanningGather(const Dimension& inputSize, const Dimension& inputStride,
        const Dimension& outputSize)
    : SimpleGatherOperation(GatherOperation::HanningGather),
      inputSize(inputSize),
      inputStride(inputStride),
      outputSize(outputSize)
    {

    }

public:
    CUDA_DECORATOR size_t operator()(size_t outputPosition) const
    {
        auto outputCoordinate = linearToDimension(outputPosition, outputSize);

        Dimension retVal = outputCoordinate;
        auto frameSize   = inputSize[0];

        retVal[0] = outputCoordinate[0] % frameSize;
        retVal[2] = outputCoordinate[2] + (outputCoordinate[0]/frameSize);

        return dotProduct(retVal, inputStride);
    }

public:
    Dimension inputSize;
    Dimension inputStride;
    Dimension outputSize;

};

class GatherIndexToOneHot: public GatherOperation
{
public:
    CUDA_DECORATOR GatherIndexToOneHot()
    : GatherIndexToOneHot(0)
    {

    }

    CUDA_DECORATOR GatherIndexToOneHot(size_t index)
    : GatherOperation(GatherOperation::GatherIndexToOneHot),
      index(index)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperation(const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& inputView) const
    {
        Dimension inputDimension;

        for(size_t dimension = 0; dimension < outputDimension.size() + 1; ++dimension)
        {
            if(index != dimension)
            {
                inputDimension.push_back(outputDimension[dimension]);
            }
            else
            {
                inputDimension.push_back(0);
            }
        }

        NativeType indexValue = inputView(inputDimension);

        if(indexValue == outputDimension[index])
        {
            return NativeType(1);
        }
        else
        {
            return NativeType(0);
        }
    }

private:
    size_t index;
};

class MapOutputToIndexDimension: public GatherOperation
{
public:
    CUDA_DECORATOR MapOutputToIndexDimension()
    : GatherOperation(GatherOperation::MapOutputToIndexDimension)
    {}

    CUDA_DECORATOR MapOutputToIndexDimension(const Dimension& indexDimensionOrder,
        size_t indexPosition,
        const Dimension& inputDimensionOrder)
    : GatherOperation(GatherOperation::MapOutputToIndexDimension),
       indexDimensionOrder(indexDimensionOrder),
       indexPosition(indexPosition),
       inputDimensionOrder(inputDimensionOrder)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperation(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView,
        const ConstMatrixView<NativeType>& indexView) const
    {
        Dimension indexDimension;

        for(size_t i = 0; i < indexDimensionOrder.size(); ++i)
        {
            indexDimension.push_back(outputDimension[indexDimensionOrder[i]]);
        }

        size_t indexValue = indexView(indexDimension);

        Dimension inputDimension;

        for(size_t i = 0, j = 0; i < inputDimensionOrder.size() + 1; ++i)
        {
            if(i == indexPosition)
            {
                inputDimension.push_back(indexValue);
            }
            else
            {
                inputDimension.push_back(outputDimension[inputDimensionOrder[j++]]);
            }
        }

        return inputView(inputDimension);
    }

public:
    Dimension indexDimensionOrder;
    size_t    indexPosition;
    Dimension inputDimensionOrder;

};

class MapOutputToMatchingIndexDimension: public GatherOperation
{
public:
    CUDA_DECORATOR MapOutputToMatchingIndexDimension()
    : GatherOperation(GatherOperation::MapOutputToMatchingIndexDimension)
    {}

    CUDA_DECORATOR MapOutputToMatchingIndexDimension(const Dimension& indexDimensionOrder,
        size_t indexPosition,
        const Dimension& inputDimensionOrder)
    : GatherOperation(GatherOperation::MapOutputToMatchingIndexDimension),
       indexDimensionOrder(indexDimensionOrder),
       indexPosition(indexPosition),
       inputDimensionOrder(inputDimensionOrder)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperation(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView,
        const ConstMatrixView<NativeType>& indexView) const
    {
        Dimension indexDimension;

        for(size_t i = 0; i < indexDimensionOrder.size(); ++i)
        {
            indexDimension.push_back(outputDimension[indexDimensionOrder[i]]);
        }

        size_t indexValue = indexView(indexDimension);

        Dimension inputDimension;

        for(size_t i = 0, j = 0; i < inputDimensionOrder.size() + 1; ++i)
        {
            if(i == indexPosition)
            {
                inputDimension.push_back(indexValue);
            }
            else
            {
                inputDimension.push_back(outputDimension[inputDimensionOrder[j++]]);
            }
        }

        return inputView(inputDimension);
    }

public:
    Dimension indexDimensionOrder;
    size_t    indexPosition;
    Dimension inputDimensionOrder;

};

typedef std::tuple<Pool2DGather, Pool2DGatherInverse, PermuteDimensionGather, HanningGather,
                   GatherIndexToOneHot> AllGatherOperations;

typedef std::tuple<MapOutputToIndexDimension, MapOutputToMatchingIndexDimension>
                   AllIndirectGatherOperations;

}
}


