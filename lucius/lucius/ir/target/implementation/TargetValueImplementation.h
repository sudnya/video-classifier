/*  \file   TargetValueImplementation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the TargetValueImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

#include <lucius/ir/target/implementation/TargetValueImplementation.h>

#include <lucius/ir/target/interface/TargetValueData.h>

// Forward Declarations
namespace lucius { namespace ir { class Use; } }

namespace lucius
{

namespace ir
{

class TargetValueImplementation : public ValueImplementation
{
public:
    using UseList = std::list<Use>;

public:
    UseList& getDefinitions();
    const UseList& getDefinitions() const;

public:
    TargetValueData getData() const;
    void setData(TargetValueData d);

public:
    virtual void allocateData() = 0;
    virtual void freeData()  = 0;

private:
    UseList _definitions;

private:
    TargetValueData _data;
};

} // namespace ir
} // namespace lucius





