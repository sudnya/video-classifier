/*  \file   ScalarOperations.h
    \date   April 17, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for common operations on scalars.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace matrix { class Scalar;         } }
namespace lucius { namespace matrix { class StaticOperator; } }

namespace lucius
{
namespace matrix
{

void apply(Scalar& result, const Scalar& left, const Scalar& right, const StaticOperator& op);
Scalar apply(const Scalar& left, const Scalar& right, const StaticOperator& op);

void apply(Scalar& result, const Scalar& input, const StaticOperator& op);
Scalar apply(const Scalar& input, const StaticOperator& op);

}
}


