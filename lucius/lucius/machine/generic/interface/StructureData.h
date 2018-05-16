/*  \file   StructureData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the StructureData class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }
namespace lucius { namespace ir { class TargetValueData;               } }

namespace lucius { namespace machine { namespace generic { class StructureDataImplementation; } } }

namespace lucius
{
namespace machine
{
namespace generic
{


/*! \brief A class for representing a structure. */
class StructureData
{
public:
    using DataVector = std::vector<ir::TargetValueData>;

public:
    explicit StructureData(const DataVector& types);
    StructureData(std::shared_ptr<ir::TargetValueDataImplementation> implementation);

public:
    ir::TargetValueData operator[](size_t );

public:
    std::shared_ptr<ir::TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<StructureDataImplementation> _implementation;

};

} // namespace generic
} // namespace machine
} // namespace lucius











