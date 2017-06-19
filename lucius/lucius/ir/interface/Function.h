/*  \file   Function.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Function class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class FunctionImplementation; } }
namespace lucius { namespace ir { class BasicBlock;             } }

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
    BasicBlock addBasicBlock();

public:
    void setIsInitializer(bool isInitializer);

private:
    std::shared_ptr<FunctionImplementation> _implementation;
};

} // namespace ir
} // namespace lucius





