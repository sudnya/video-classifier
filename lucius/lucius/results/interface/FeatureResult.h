/*    \file   FeatureResult.h
    \date   Saturday August 10, 2014
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the FeatureResult class.
*/

#pragma once

// Lucius Includes
#include <lucius/results/interface/Result.h>

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
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



