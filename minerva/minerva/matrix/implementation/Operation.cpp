/*    \file   Operation.cpp
    \date   Sunday April 11, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the Operation classes.
*/

// Lucious Includes
#include <lucious/matrix/interface/Operation.h>

namespace lucious
{
namespace matrix
{

bool Operation::operator==(const Operation& o) const
{
    return o._type == _type;
}

}
}

