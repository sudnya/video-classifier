/*  \file   ConstantOperators.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantOperators class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

// Forward Declaration
namespace lucius { namespace matrix { class Operator; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant matrix value a program. */
class ConstantOperator : public Constant
{
public:
    explicit ConstantOperator(const matrix::Operator&);
    explicit ConstantOperator(std::shared_ptr<ValueImplementation>);

public:
    const matrix::Operator& getOperator() const;
};

class Add : public ConstantOperator
{
public:
    Add();

};

class Multiply : public ConstantOperator
{
public:
    Multiply();

};

} // namespace ir
} // namespace lucius








