/*  \file   CudaTargetOperationFactory.cpp
    \author Gregory Diamos
    \date   August 16, 2017
    \brief  The source file for the CudaTargetOperationFactory class.
*/

// Lucius Includes
#include <lucius/machine/cuda/interface/CudaTargetOperationFactory.h>

#include <lucius/machine/generic/interface/TensorValue.h>

#include <lucius/ir/interface/Type.h>

#include <lucius/ir/types/interface/TensorType.h>

#include <lucius/ir/target/interface/TargetOperation.h>
#include <lucius/ir/target/interface/TargetValue.h>

namespace lucius
{

namespace machine
{

namespace cuda
{

CudaTargetOperationFactory::CudaTargetOperationFactory(Context& context)
: TargetOperationFactory(context)
{

}

CudaTargetOperationFactory::~CudaTargetOperationFactory()
{
    // intentionally blank
}

ir::TargetOperation CudaTargetOperationFactory::create(const std::string& name) const
{
    throw std::runtime_error("No support for creating operation named '" + name +
        "' for the CUDA target machine.");
}

ir::TargetValue CudaTargetOperationFactory::createOperand(const ir::Type& t) const
{
    if(t.isTensor())
    {
        auto tensorType = ir::type_cast<ir::TensorType>(t);

        return generic::TensorValue(tensorType.getShape(), tensorType.getPrecision(), _context);
    }

    throw std::runtime_error("No support for creating operand of type '" + t.toString() +
        "' for CPU target machine.");
}

} // namespace cuda
} // namespace machine
} // namespace lucius






