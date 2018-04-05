/*  \file   StaticOperator.h
    \date   October 30, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the StaticOperator classes.
*/

#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/ScalarOperations.h>

#include <lucius/matrix/interface/DimensionTransformations.h>
#include <lucius/matrix/interface/MatrixView.h>
#include <lucius/matrix/interface/StaticOperator.h>

// Standard Library Includes
#include <cmath>
#include <algorithm>

namespace lucius
{
namespace matrix
{

class Pool2DGather : public StaticOperator
{
public:
    CUDA_DECORATOR Pool2DGather()
    : StaticOperator(StaticOperator::Pool2DGather)
    {}

    CUDA_DECORATOR Pool2DGather(size_t w, size_t h, size_t inputW, size_t inputH)
    : StaticOperator(StaticOperator::Pool2DGather),
      width(w), height(h), inputWidth(inputW), inputHeight(inputH)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView) const
    {
        size_t outputIndex = dotProduct(outputDimension, outputView.stride());

        size_t inputIndex = (*this)(outputIndex);

        return inputView(linearToDimension(inputIndex, inputView.size()));
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
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Pool2DGather>(*this);
    }

public:
    size_t width;
    size_t height;

    size_t inputWidth;
    size_t inputHeight;

};

class Pool2DGatherInverse : public StaticOperator
{
public:
    CUDA_DECORATOR Pool2DGatherInverse()
    : StaticOperator(StaticOperator::Pool2DGatherInverse)
    {}

    CUDA_DECORATOR Pool2DGatherInverse(size_t w, size_t h, size_t inputW, size_t inputH)
    : StaticOperator(StaticOperator::Pool2DGatherInverse),
      width(w), height(h), inputWidth(inputW), inputHeight(inputH)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView) const
    {
        size_t outputIndex = dotProduct(outputDimension, outputView.stride());

        size_t inputIndex = (*this)(outputIndex);

        return inputView(linearToDimension(inputIndex, inputView.size()));
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
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<Pool2DGatherInverse>(*this);
    }

public:
    size_t width;
    size_t height;

    size_t inputWidth;
    size_t inputHeight;

};

class PermuteDimensionGather : public StaticOperator
{
public:
    CUDA_DECORATOR PermuteDimensionGather()
    : StaticOperator(StaticOperator::PermuteDimensionGather)
    {}

    CUDA_DECORATOR PermuteDimensionGather(const Dimension& inputStride,
        const Dimension& outputSize, const Dimension& order)
    : StaticOperator(StaticOperator::PermuteDimensionGather), inputStride(inputStride),
      outputSize(outputSize), order(order)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView) const
    {
        size_t outputIndex = dotProduct(outputDimension, outputView.stride());

        size_t inputIndex = (*this)(outputIndex);

        return inputView(linearToDimension(inputIndex, inputView.size()));
    }

public:
    CUDA_DECORATOR size_t operator()(size_t outputPosition) const
    {
        auto outputDimension = linearToDimension(outputPosition, outputSize);

        auto inputDimension = selectReverseMappingDimensions(outputDimension, order);

        return dotProduct(inputDimension, inputStride);
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<PermuteDimensionGather>(*this);
    }

public:
    Dimension inputStride;
    Dimension outputSize;
    Dimension order;

};

class HanningGather : public StaticOperator
{
public:
    CUDA_DECORATOR HanningGather()
    : StaticOperator(StaticOperator::HanningGather)
    {}

    CUDA_DECORATOR HanningGather(const Dimension& inputSize, const Dimension& inputStride,
        const Dimension& outputSize)
    : StaticOperator(StaticOperator::HanningGather),
      inputSize(inputSize),
      inputStride(inputStride),
      outputSize(outputSize)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(
        const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView) const
    {
        size_t outputIndex = dotProduct(outputDimension, outputView.stride());

        size_t inputIndex = (*this)(outputIndex);

        return inputView(linearToDimension(inputIndex, inputView.size()));
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
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<HanningGather>(*this);
    }

public:
    Dimension inputSize;
    Dimension inputStride;
    Dimension outputSize;

};

class GatherIndexToOneHot: public StaticOperator
{
public:
    CUDA_DECORATOR GatherIndexToOneHot()
    : GatherIndexToOneHot(0)
    {

    }

    CUDA_DECORATOR GatherIndexToOneHot(size_t index)
    : StaticOperator(StaticOperator::GatherIndexToOneHot),
      index(index)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(const Dimension& outputDimension,
        const ConstMatrixView<NativeType>& outputView,
        const ConstMatrixView<NativeType>& inputView) const
    {
        Dimension inputDimension;

        for(size_t dimension = 0; dimension < outputDimension.size(); ++dimension)
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

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<GatherIndexToOneHot>(*this);
    }

private:
    size_t index;
};

class MapOutputToIndexDimension: public StaticOperator
{
public:
    CUDA_DECORATOR MapOutputToIndexDimension()
    : StaticOperator(StaticOperator::MapOutputToIndexDimension)
    {}

    CUDA_DECORATOR MapOutputToIndexDimension(const Dimension& indexDimensionOrder,
        size_t indexPosition,
        const Dimension& inputDimensionOrder)
    : StaticOperator(StaticOperator::MapOutputToIndexDimension),
       indexDimensionOrder(indexDimensionOrder),
       indexPosition(indexPosition),
       inputDimensionOrder(inputDimensionOrder)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(
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
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<MapOutputToIndexDimension>(*this);
    }

public:
    Dimension indexDimensionOrder;
    size_t    indexPosition;
    Dimension inputDimensionOrder;

};

class MapOutputToMatchingIndexDimension: public StaticOperator
{
public:
    CUDA_DECORATOR MapOutputToMatchingIndexDimension()
    : StaticOperator(StaticOperator::MapOutputToMatchingIndexDimension)
    {}

    CUDA_DECORATOR MapOutputToMatchingIndexDimension(const Dimension& indexDimensionOrder,
        const Dimension& inputDimensionOrder,
        size_t matchingOutputPosition,
        double defaultValue)
    : StaticOperator(StaticOperator::MapOutputToMatchingIndexDimension),
      indexDimensionOrder(indexDimensionOrder),
      inputDimensionOrder(inputDimensionOrder),
      matchingOutputPosition(matchingOutputPosition),
      defaultValue(defaultValue)
    {

    }

public:
    template <typename NativeType>
    CUDA_DECORATOR NativeType runOperator(
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

        size_t outputIndexValue = outputDimension[matchingOutputPosition];

        NativeType resultValue = defaultValue;

        if(indexValue == outputIndexValue)
        {
            Dimension inputDimension;

            for(size_t i = 0; i < indexDimensionOrder.size(); ++i)
            {
                inputDimension.push_back(outputDimension[inputDimensionOrder[i]]);
            }

            resultValue = inputView(inputDimension);
        }

        return resultValue;
    }

public:
    std::unique_ptr<StaticOperator> clone() const
    {
        return std::make_unique<MapOutputToMatchingIndexDimension>(*this);
    }

public:
    Dimension indexDimensionOrder;
    Dimension inputDimensionOrder;

    size_t    matchingOutputPosition;
    double    defaultValue;

};

typedef std::tuple<Pool2DGather, Pool2DGatherInverse, PermuteDimensionGather, HanningGather,
                   GatherIndexToOneHot> AllGatherOperators;

typedef std::tuple<MapOutputToIndexDimension, MapOutputToMatchingIndexDimension>
                   AllIndirectGatherOperators;

}
}


