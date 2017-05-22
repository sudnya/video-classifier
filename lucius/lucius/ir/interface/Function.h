/*  \file   Function.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Function class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a function. */
class Function
{
public:
    Function();
    ~Function();

public:
    BasicBlock* addBasicBlock();

public:
    void setIsInitializer(bool isInitializer);

private:
    Module* _module;

private:
    BasicBlockList _basicBlocks;
    ArgumentList   _arguments;

};

} // namespace ir
} // namespace lucius





