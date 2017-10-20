/*  \file   OperationFactory.cpp
    \author Gregory Diamos
    \date   October 11, 2017
    \brief  The source file for the OperationFactory class.
*/

// Lucius Includes
#include <lucius/ir/ops/interface/OperationFactory.h>

// Standard Library Includes
#include <map>

namespace lucius
{

namespace ir
{

class OperationFactoryImplementation
{
public:
    Operation createBackPropagationOperation(const Operation& op)
    {
        auto backPropagationOp = _backPropagationOperations.find(op);

        if(backPropagationOp == _backPropagationOperations.end())
        {
            throw std::runtime_error("There is no registered back propagation operation for '" +
                op.name() + "'");
        }

        return backPropagationOp->second;;
    }

public:
    void addBackPropagationOperation(const Operation& original, const Operation& back)
    {
        _backPropagationOperations.emplace(std::make_pair(original, back));
    }

private:
    std::map<Operation, Operation> _backPropagationOperations;

};

static std::unique_ptr<OperationFactoryImplementation> implementation;

static OperationFactoryImplementation& getImplementation()
{
    if(!implementation)
    {
        implementation = std::make_unique<OperationFactoryImplementation>();
    }

    return *implementation;
}

Operation OperationFactory::createBackPropagationOperation(const Operation& op)
{
    return getImplementation().createBackPropagationOperation(op);
}

} // namespace ir
} // namespace lucius




