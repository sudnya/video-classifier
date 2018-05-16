/*  \file   StructureData.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the StructureData class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/StructureData.h>

#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/ir/target/implementation/TargetValueDataImplementation.h>

// Standard Library Includes
#include <vector>

namespace lucius
{
namespace machine
{
namespace generic
{

class StructureDataImplementation : public ir::TargetValueDataImplementation
{
public:
    StructureDataImplementation(const StructureData::DataVector& data)
    : _members(data)
    {
    }

public:
    ir::TargetValueData operator[](size_t position)
    {
        return _members[position];
    }

public:
    virtual void* getData() const
    {
        return _members.front().data();
    }

private:
    std::vector<ir::TargetValueData> _members;

};

StructureData::StructureData(const DataVector& data)
: StructureData(std::make_shared<StructureDataImplementation>(data))
{

}

StructureData::StructureData(std::shared_ptr<ir::TargetValueDataImplementation> implementation)
: _implementation(std::static_pointer_cast<StructureDataImplementation>(implementation))
{
    assert(static_cast<bool>(std::dynamic_pointer_cast<StructureDataImplementation>(
        implementation)));
}

ir::TargetValueData StructureData::operator[](size_t offset)
{
    return (*_implementation)[offset];
}

std::shared_ptr<ir::TargetValueDataImplementation> StructureData::getImplementation() const
{
    return _implementation;
}

} // namespace generic
} // namespace machine
} // namespace lucius











