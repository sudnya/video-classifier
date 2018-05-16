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

namespace lucius { namespace machine { namespace generic { class OperatorDataImplementation; } } }

namespace lucius { namespace matrix { class Operator; } }

namespace lucius
{
namespace machine
{
namespace generic
{

/*! \brief A class for representing an operator value resource. */
class OperatorData
{
public:
    OperatorData();
    OperatorData(std::shared_ptr<ir::TargetValueDataImplementation> implementation);

public:
    matrix::Operator getOperator() const;

public:
    std::shared_ptr<ir::TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<OperatorDataImplementation> _implementation;

};

} // namespace generic
} // namespace machine
} // namespace lucius









