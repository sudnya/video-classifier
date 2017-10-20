/*  \file   ComputeGradientOperationImplementation.h
    \author Gregory Diamos
    \date   October 12, 2017
    \brief  The header file for the ComputeGradientOperationImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/OperationImplementation.h>

namespace lucius
{

namespace ir
{

class ComputeGradientOperationImplementation : public OperationImplementation
{
public:
    ShapeList getOutputShapes(const ShapeList& inputShapes) const;
    ShapeList getInputShapes(const ShapeList& outputShapes) const;

};

} // namespace ir
} // namespace lucius

