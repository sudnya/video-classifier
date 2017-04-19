/*  \file   Module.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Module class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a module. */
class Module
{

private:
    ConstantMap _constants;
    FunctionMap _functions;

};

} // namespace ir
} // namespace lucius




