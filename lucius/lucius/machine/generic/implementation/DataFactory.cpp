/*  \file   DataFactory.cpp
    \author Gregory Diamos
    \date   April 24, 2018
    \brief  The header file for the DataFactory class.
*/

// Lucius Includes
#include <lucius/machine/generic/interface/DataFactory.h>

#include <lucius/ir/interface/Type.h>

#include <lucius/ir/types/interface/StructureType.h>
#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/interface/TargetValueData.h>

#include <lucius/machine/generic/interface/FloatData.h>
#include <lucius/machine/generic/interface/IntegerData.h>
#include <lucius/machine/generic/interface/PointerData.h>
#include <lucius/machine/generic/interface/RandomStateData.h>
#include <lucius/machine/generic/interface/StructureData.h>
#include <lucius/machine/generic/interface/TensorData.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <vector>

namespace lucius
{
namespace machine
{
namespace generic
{

ir::TargetValueData DataFactory::create(const ir::Type& type)
{
    if(type.isInteger())
    {
        return IntegerData();
    }
    else if(type.isFloat())
    {
        return FloatData();
    }
    else if(type.isRandomState())
    {
        return RandomStateData();
    }
    else if(type.isPointer())
    {
        return PointerData();
    }
    else if(type.isTensor())
    {
        auto tensor = ir::type_cast<ir::TensorType>(type);

        return TensorData(tensor.getShape(), tensor.getPrecision());
    }
    else if(type.isStructure())
    {
        std::vector<ir::TargetValueData> members;

        auto structureType = ir::type_cast<ir::StructureType>(type);

        for(auto& memberType : structureType)
        {
            members.push_back(create(memberType));
        }

        return StructureData(members);
    }

    assertM(false, "Type not supported.");

    return ir::TargetValueData();
}

} // namespace generic
} // namespace machine
} // namespace lucius



