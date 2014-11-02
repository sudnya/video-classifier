/*	\file   FeatureResult.h
	\date   Saturday August 10, 2014
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the FeatureResult class.
*/

#pragma once

// Minerva Includes
#include <minerva/results/interface/Result.h>

// Forward Declarations
namespace minerva { namespace matrix { class Matrix; } }

namespace minerva
{

namespace results
{

/*! \brief The label assigned to a sample.  */
class FeatureResult : public Result
{
public:
	typedef matrix::Matrix Matrix;

public:
	FeatureResult(Matrix&&);
	~FeatureResult();
	
public:
	std::unique_ptr<Matrix> features;

};

}

}



