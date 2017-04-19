/*  \file   Operation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Operation class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation. */
class Operation
{
public:
    Operation(const std::string& name, const ArgumentList& inputs, const ArgumentList& outputs);
    virtual Operation();

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

private:
    // operation name
    std::string _name;

private:
    // input values
    ArgumentList _inputs;

    // output values
    ArgumentList _outputs;

};

} // namespace ir
} // namespace lucius



