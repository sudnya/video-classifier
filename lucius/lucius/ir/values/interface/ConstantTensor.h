/*  \file   ConstantTensor.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ConstantTensor class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Constant.h>

// Forward Declarations
namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a constant matrix value a program. */
class ConstantTensor : public Constant
{
public:
    explicit ConstantTensor(const matrix::Matrix& value);
};

} // namespace ir
} // namespace lucius






