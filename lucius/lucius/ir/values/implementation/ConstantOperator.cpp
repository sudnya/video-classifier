/*  \file   ConstantOperators.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the ConstantOperators class.
*/

// Lucius Includes
#include <lucius/ir/values/interface/ConstantOperator.h>

#include <lucius/ir/implementation/ConstantImplementation.h>

namespace lucius
{

namespace ir
{

enum OperatorIds
{
    Unknown = 0,
    AddId = 1
};

class ConstantOperatorValueImplementation : public ConstantImplementation
{
public:
    ConstantOperatorValueImplementation(size_t id)
    : _id(id)
    {

    }

    size_t getId() const
    {
        return _id;
    }

private:
    size_t _id;
};

ConstantOperator::ConstantOperator(size_t operatorId)
: Constant(std::make_shared<ConstantOperatorValueImplementation>(operatorId))
{

}

Add::Add()
: ConstantOperator(AddId)
{

}

} // namespace ir
} // namespace lucius









