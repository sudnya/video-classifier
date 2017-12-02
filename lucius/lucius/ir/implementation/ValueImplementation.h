/*  \file   ValueImplementation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ValueImplementation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Type.h>

// Standard Library Includes
#include <list>

// Forward Declarations
namespace lucius { namespace ir { class Use; } }

namespace lucius
{

namespace ir
{

/*! \brief The implementation of a class that represents a value in the program.

    Aside: Inheritance is the base class of evil.
*/
class ValueImplementation
{
public:
    ValueImplementation();
    virtual ~ValueImplementation();

public:
          Type& getType();
    const Type& getType() const;

public:
    typedef std::list<Use> UseList;

public:
          UseList& getUses();
    const UseList& getUses() const;

public:
    size_t getId() const;

public:
    bool isOperation() const;
    bool isConstant() const;

public:
    bool isVariable() const;
    void setIsVariable(bool b);

public:
    virtual std::shared_ptr<ValueImplementation> clone() const = 0;

private:
    size_t _id;

private:
    Type _type;

private:
    UseList _uses;

private:
    bool _isVariable;

};

} // namespace ir
} // namespace lucius




