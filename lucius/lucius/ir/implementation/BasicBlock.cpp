/*  \file   BasicBlock.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the BasicBlock class.
*/

// Lucius Includes
#include <lucius/ir/interface/BasicBlock.h>

#include <lucius/ir/interface/Function.h>
#include <lucius/ir/interface/Value.h>

namespace lucius
{

namespace ir
{

class BasicBlockImplementation : public Value
{
private:
    std::weak_ptr<FunctionImplementation> _parent;

private:
    using BasicBlockList = std::list<BasicBlock>;

private:
    BasicBlockList::iterator _position;

private:
    using OperationList = std::list<Operation>;

private:
    OperationList _operations;

};

} // namespace ir
} // namespace lucius





