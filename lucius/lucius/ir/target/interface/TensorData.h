/*  \file   TensorData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the TensorData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }
namespace lucius { namespace ir { class TensorDataImplementation;      } }

namespace lucius { namespace matrix { class Matrix; } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a tensor value resource. */
class TensorData
{
public:
    TensorData();
    TensorData(std::shared_ptr<TargetValueDataImplementation> implementation);

public:
    matrix::Matrix getTensor() const;

public:
    std::shared_ptr<TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<TensorDataImplementation> _implementation;

};

} // namespace ir
} // namespace lucius








