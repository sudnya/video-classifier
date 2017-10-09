/*  \file   IRBuilder.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRBuilder class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlock;                   } }
namespace lucius { namespace ir { class Operation;                    } }
namespace lucius { namespace ir { class Function;                     } }
namespace lucius { namespace ir { class InsertionPointImplementation; } }

namespace lucius
{
namespace ir
{

class InsertionPoint
{
public:
    InsertionPoint(const BasicBlock& block);
    InsertionPoint(const Operation& operation);
    InsertionPoint();
    InsertionPoint(const InsertionPoint&);
    ~InsertionPoint();

public:
    InsertionPoint& operator=(const InsertionPoint&);

public:
    using BasicBlockList = std::list<BasicBlock>;

public:
    BasicBlock& getBasicBlock();
    BasicBlockList::iterator& getIterator();
    Function& getFunction();

public:
    std::unique_ptr<InsertionPointImplementation> _implementation;

};

} // namespace ir
} // namespace lucius

