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

/*! \brief A class for representing an operation. */
class Operation : public User
{
public:
    Operation(const std::string& name, const ArgumentList& inputs, const ArgumentList& outputs);
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
    // operation name
    std::string _name;

private:
    // operands
    UseList _operands;

};

typedef std::list<std::unique_ptr<Operation>> OperationList;

} // namespace ir
} // namespace lucius



