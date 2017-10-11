/*  \file   RandomStateType.cpp
    \author Gregory Diamos
    \date   October 4, 2017
    \brief  The source file for the RandomStateType class.
*/

// Lucius Includes
#include <lucius/ir/types/interface/RandomStateType.h>

#include <lucius/ir/implementation/TypeImplementation.h>

namespace lucius
{

namespace ir
{

class RandomStateTypeImplementation : public TypeImplementation
{
public:
    RandomStateTypeImplementation()
    : TypeImplementation(Type::StructureId)
    {

    }

};

RandomStateType::RandomStateType()
: Type(std::make_shared<RandomStateTypeImplementation>())
{

}

RandomStateType::~RandomStateType()
{

}

} // namespace ir
} // namespace lucius





