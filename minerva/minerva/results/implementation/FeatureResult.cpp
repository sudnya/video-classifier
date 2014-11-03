/*	\file   FeatureResult.cpp
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the FeatureResult class.
*/

// Minerva Includes
#include <minerva/results/interface/FeatureResult.h>
#include <minerva/matrix/interface/Matrix.h>

namespace minerva
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




