/*  \file   StaticOperator.cpp
    \date   Sunday April 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the StaticOperator classes.
*/

// Lucius Includes
#include <lucius/matrix/interface/StaticOperator.h>

namespace lucius
{
namespace matrix
{

bool StaticOperator::operator==(const StaticOperator& o) const
{
    return o._type == _type;
}

}
}

