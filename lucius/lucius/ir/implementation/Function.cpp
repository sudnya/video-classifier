/*  \file   Function.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Function class.
*/

// Lucius Includes
#include <lucius/ir/interface/Function.h>

#include <lucius/ir/interface/Value.h>
#include <lucius/ir/interface/Module.h>

namespace lucius
{

namespace ir
{

class FunctionImplementation : public Value
{

private:
    using FunctionList = std::list<Function>;

private:
    FunctionList::iterator _position;

private:
    Module _parent;

private:
    using BasicBlockList = std::list<BasicBlock>;

private:
    BasicBlockList _blocks;

};

} // namespace ir
} // namespace lucius






