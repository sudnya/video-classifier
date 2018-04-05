/*  \file   PointerData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the PointerData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }
namespace lucius { namespace ir { class PointerDataImplementation;    } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a tensor value resource. */
class PointerData
{
public:
    PointerData();
    PointerData(void* value);
    PointerData(std::shared_ptr<TargetValueDataImplementation> implementation);

public:
    void* getPointer() const;

public:
    std::shared_ptr<TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<PointerDataImplementation> _implementation;

};

} // namespace ir
} // namespace lucius











