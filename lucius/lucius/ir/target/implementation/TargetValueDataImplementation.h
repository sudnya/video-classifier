/*  \file   TargetValueDataImplementation.h
    \author Gregory Diamos
    \date   December 19, 2017
    \brief  The header file for the TargetValueDataImplementation class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a managed data resource. */
class TargetValueDataImplementation
{
public:
    virtual ~TargetValueDataImplementation();

public:
    virtual void* getData() const = 0;
};

} // namespace ir
} // namespace lucius








