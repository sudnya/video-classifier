/*  \file   Operation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/User.h>

// Standard Library Includes
#include <string>
#include <vector>

namespace lucius
{

namespace ir
{

class OperationImplementation : public User
{
private:
    std::weak_ptr<BasicBlockImplementation> _parent;

};

/*! \brief A class for representing an operation. */
class Operation
{
public:
    Operation(const ArgumentList& inputs, const ArgumentList& outputs);
    virtual ~Operation();

public:
    // forward operation
    virtual void runForwardProagation() = 0;

    // backward inputs operation
    virtual void runBackwardPropagation() = 0;

public:
    // forward shape operation
    virtual ShapeList getOutputShapes(const ShapeList& inputShapes) const = 0;

    // backward shape operation
    virtual ShapeList getInputShapes(const ShapeList& outputShapes) const = 0;

public:
    // forward performance metrics
    virtual PerformanceMetrics getForwardPerformanceMetrics(
        const ShapeList& inputShapes) const = 0;

    // backward performance metrics
    virtual PerformanceMetrics getBackwardPerformanceMetrics(
        const ShapeList& outputShapes) const = 0;

public:
    const UseList& getOperands() const;
          UseList& getOperands();

private:
    std::shared_ptr<OperationImplementation> _implementation;

};

} // namespace ir
} // namespace lucius



