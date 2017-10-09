/*  \file   CopyOperation.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the CopyOperation class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/CopyOperation.h>

#include <lucius/ir/interface/ShapeList.h>

#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class CopyOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const
    {
        return inputShapes;
    }

    ShapeList getInputShapes(const ShapeList& outputShapes) const
    {
        return outputShapes;
    }

};

CopyOperation::CopyOperation(Value input)
: Operation(std::make_shared<CopyOperationImplementation>())
{
    setOperands({input});
}

CopyOperation::~CopyOperation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius


