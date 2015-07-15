/*	\file   FeatureResult.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureResult class.
*/

// Lucius Includes
#include <lucius/results/interface/FeatureResult.h>
#include <lucius/matrix/interface/Matrix.h>

namespace lucius
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




