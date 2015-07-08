/*	\file   FeatureResult.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureResult class.
*/

// Lucious Includes
#include <lucious/results/interface/FeatureResult.h>
#include <lucious/matrix/interface/Matrix.h>

namespace lucious
{

namespace results
{

FeatureResult::FeatureResult(Matrix&& m)
: features(new Matrix(std::move(m)))
{

}

FeatureResult::~FeatureResult()
{

}

}

}




