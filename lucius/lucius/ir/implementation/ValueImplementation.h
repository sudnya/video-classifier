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
namespace lucius { namespace ir { class Use;     } }
namespace lucius { namespace ir { class Context; } }

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
    typedef std::list<Use> UseList;

public:
          UseList& getUses();
    const UseList& getUses() const;

public:
    size_t getId() const;

public:
    bool isOperation() const;
    bool isTargetOperation() const;
    bool isConstant() const;
    bool isFunction() const;
    bool isExternalFunction() const;
    bool isTargetValue() const;

public:
    bool isVariable() const;
    void setIsVariable(bool b);

public:
    void bindToContext(Context* context);
    void bindToContextIfDifferent(Context* context);
    Context* getContext();

public:
    virtual std::shared_ptr<ValueImplementation> clone() const = 0;

public:
    virtual std::string name() const;
    virtual std::string toString() const = 0;
    virtual std::string toSummaryString() const;

public:
    virtual Type getType() const = 0;

public:
    virtual bool isCall() const;
    virtual bool isReturn() const;
    virtual bool isPHI() const;

private:
    size_t _id;

private:
    UseList _uses;

private:
    bool _isVariable;

private:
    Context* _context;

};

} // namespace ir
} // namespace lucius




