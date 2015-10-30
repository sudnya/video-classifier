/*! \file   math.h
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Friday March 15, 2014
    \brief  The header file for the library of math utility functions.
*/

#pragma once

// Standard Library Includes
#include <cstring>

namespace lucius
{

namespace util
{

/*! \brief Get the square root of a number (integer) */
size_t getSquareRoot(size_t number);

/*! \brief Get the nearest to square factors of a number (e.g. 8, 5 for 40) */
void getNearestToSquareFactors(size_t& factor1, size_t& factor2, size_t number);

}

}

