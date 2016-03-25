/*! \file EditDistance.h
    \date March 21, 2016
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief Function headers for common edit distance metrics
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace lucius
{

namespace util
{

typedef std::vector<std::string> StringVector;

size_t editDistance(const std::string& left, const std::string& right);

size_t editDistance(const std::string& left, const std::string& right,
    const StringVector& graphemes);

}

}


