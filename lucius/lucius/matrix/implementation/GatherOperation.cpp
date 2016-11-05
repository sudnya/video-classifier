/*  \file   GatherOperation.cpp
    \date   October 30, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the GatherOperation classes.
*/

// Lucius Includes
#include <lucius/matrix/interface/GatherOperation.h>

namespace lucius
{
namespace matrix
{


CUDA_DECORATOR bool GatherOperation::operator==(const GatherOperation& op) const
{
    return _type == op._type;
}

}
}



