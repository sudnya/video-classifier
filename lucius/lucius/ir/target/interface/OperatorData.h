/*  \file   OperatorData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the OperatorData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }
namespace lucius { namespace ir { class OperatorDataImplementation;    } }

namespace lucius { namespace matrix { class Operator; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a tensor value resource. */
class OperatorData
{
public:
    OperatorData();
    OperatorData(std::shared_ptr<TargetValueDataImplementation> implementation);

public:
    matrix::Operator getOperator() const;

public:
    std::shared_ptr<TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<OperatorDataImplementation> _implementation;

};

} // namespace ir
} // namespace lucius









