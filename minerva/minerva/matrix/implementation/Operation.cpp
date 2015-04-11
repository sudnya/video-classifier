/*	\file   Operation.cpp
	\date   Sunday April 11, 2015
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Operation classes.
*/

// Minerva Includes
#include <minerva/matrix/interface/Operation.h>

namespace minerva
{
namespace matrix
{

Operation::Operation(Type t)
: _type(t)
{

}

bool Operation::operator==(const Operation& o) const
{
	return o._type == _type;
}

}
}

