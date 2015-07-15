/*    \file   Operation.cpp
    \date   Sunday April 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Operation classes.
*/

// Lucius Includes
#include <lucius/matrix/interface/Operation.h>

namespace lucius
{
namespace matrix
{

bool Operation::operator==(const Operation& o) const
{
    return o._type == _type;
}

}
}

