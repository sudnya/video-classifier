/*  \file   OperationFactory.h
    \author Gregory Diamos
    \date   August 4, 2017
    \brief  The header file for the OperationFactory class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for creating operations. */
class OperationFactory
{
public:
    /*! \brief Create the back propagation version of another operation. */
    static Operation createBackPropagationOperation(const Operation& op);
};

} // namespace ir
} // namespace lucius



