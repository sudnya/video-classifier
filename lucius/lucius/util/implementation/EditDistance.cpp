/*! \file EditDistance.cpp
    \date March 21, 2016
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief Function headers for common edit distance metrics
*/

// Lucius Includes
#include <lucius/util/interface/EditDistance.h>

// Standard Library Includes
#include <set>
#include <map>

namespace lucius
{

namespace util
{

static StringVector getAllCharacterGraphemes()
{
    StringVector result;

    for(size_t i = 0; i < 255; ++i)
    {
        std::string string;

        string.push_back(static_cast<char>(i));

        result.push_back(string);
    }

    return result;
}

size_t editDistance(const std::string& left, const std::string& right)
{
    return editDistance(left, right, getAllCharacterGraphemes());
}

StringVector toGraphemes(const std::string& label, const StringVector& graphemes)
{
    std::set<std::string> graphemeSet(graphemes.begin(), graphemes.end());

    auto remainingLabel = label;

    StringVector graphemeResult;

    while(!remainingLabel.empty())
    {
        auto insertPosition = graphemeSet.lower_bound(remainingLabel);

        // exact match
        if(insertPosition != graphemeSet.end() && *insertPosition == remainingLabel)
        {
            graphemeResult.push_back(remainingLabel);
            break;
        }

        // ordered before first grapheme
        if(insertPosition == graphemeSet.begin())
        {
            throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                "' against a grapheme.");
        }

        --insertPosition;

        auto grapheme = remainingLabel.substr(0, insertPosition->size());

        if(grapheme != *insertPosition)
        {
            throw std::runtime_error("Could not match remaining label '" + remainingLabel +
                "' against best grapheme '" + *insertPosition + "'.");
        }

        graphemeResult.push_back(grapheme);

        remainingLabel = remainingLabel.substr(grapheme.size());
    }

    return graphemeResult;
}

typedef std::map<std::pair<size_t, size_t>, size_t> IndexMap;

static size_t computeEditDistance(const StringVector& left, size_t leftIndex,
    const StringVector& right, size_t rightIndex, IndexMap& cachedDistances)
{
    if(leftIndex == 0)
    {
        return rightIndex;
    }
    else if(rightIndex == 0)
    {
        return leftIndex;
    }

    if(cachedDistances.count(std::make_pair(leftIndex, rightIndex)))
    {
        return cachedDistances[std::make_pair(leftIndex, rightIndex)];
    }

    size_t distance = 0;

    if(left[leftIndex] == right[rightIndex])
    {
        distance = computeEditDistance(left, leftIndex - 1, right, rightIndex - 1,
            cachedDistances);
    }
    else
    {
        size_t deleteCost = 1 + computeEditDistance(left, leftIndex, right,
            rightIndex - 1, cachedDistances);
        size_t insertCost = 1 + computeEditDistance(left, leftIndex - 1, right,
            rightIndex - 1, cachedDistances);
        size_t substituteCost = 1 + computeEditDistance(left, leftIndex - 1, right,
            rightIndex - 1, cachedDistances);

        distance = std::min(deleteCost, std::min(insertCost, substituteCost));
    }

    cachedDistances[std::make_pair(leftIndex, rightIndex)] = distance;

    return distance;
}

size_t editDistance(const std::string& left, const std::string& right,
    const StringVector& graphemes)
{
    StringVector leftGraphemes  = toGraphemes(left,  graphemes);
    StringVector rightGraphemes = toGraphemes(right, graphemes);

    IndexMap cachedDistances;

    return computeEditDistance(leftGraphemes, leftGraphemes.size() - 1,
        rightGraphemes, rightGraphemes.size() - 1, cachedDistances);
}

}

}



