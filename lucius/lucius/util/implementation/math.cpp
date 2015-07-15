/*! \file   math.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Friday March 15, 2014
	\brief  The source file for the library of math utility functions.
*/

// Lucius Includes
#include <lucius/util/interface/math.h>

// Standard Library Includes
#include <utility>

namespace lucius
{

namespace util
{

size_t getSquareRoot(size_t number)
{
	// start with half
	size_t potential = number / 2;

	size_t step = potential;

	// binary search
	while(true)
	{
		size_t squared = potential * potential;
		step /= 2;		

		if(squared == number)
		{
			break;
		}
		
		if(squared > number)
		{
			potential -= step;
			continue;
		}
		
		if((potential + 1) * (potential + 1) < number)
		{
			potential += step;
		}
		
		break;
	}

	return potential;
}

void getNearestToSquareFactors(size_t& factor1, size_t& factor2, size_t number)
{
	// get the integer sqrt
	size_t squareRoot = getSquareRoot(number);
	
	// start the search
	size_t upperBound = squareRoot;
	
	// linear scan down
	// TODO: more intelligent search
	for( ; upperBound > 1; --upperBound)
	{
		if(number % upperBound == 0)
		{
			break;
		}
	}
	
	factor1 = upperBound;
	factor2 = number / upperBound;

	if(factor1 < factor2)
	{
		std::swap(factor1, factor2);
	}
}

}

}

