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

/*! \brief A class for representing a module. */
class Function
{

private:
    BasicBlockList _basicBlocks;
    ArgumentList   _arguments;

};

} // namespace ir
} // namespace lucius





