/*	\file   FeatureResult.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FeatureResult class.
*/

#pragma once

#include <minerva/results/interface/Result.h>

namespace minerva
{

namespace results
{

/*! \brief The label assigned to a sample.  */
class FeatureResult : public Result
{
public:
	FeatureResult(Matrix&&);
	
public:
	Matrix features;

};

}

}



