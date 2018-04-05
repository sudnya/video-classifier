/*  \file   ExternalFunctionImplementation.h
    \author Gregory Diamos
    \date   March 17, 2018
    \brief  The header file for the ExternalFunctionImplementation class.
*/

// Lucius Includes
#include <lucius/ir/implementation/ValueImplementation.h>

#include <lucius/ir/interface/Use.h>

// Standard Library Includes
#include <vector>
#include <string>

// Forward Declarations
namespace lucius { namespace ir { class Type; } }

namespace lucius
{

namespace ir
{

class ExternalFunctionImplementation : public ValueImplementation
{
public:
    using TypeVector = std::vector<Type>;

public:
    ExternalFunctionImplementation(const std::string& name, const TypeVector& types);

public:
    const std::string& getName() const;

public:
    void setPassArgumentsAsTargetValues(bool condition);

    bool getPassArgumentsAsTargetValues() const;

public:
    virtual std::shared_ptr<ValueImplementation> clone() const;
    virtual std::string toString() const;
    virtual std::string toSummaryString() const;
    virtual Type getType() const;

private:
    std::string _name;

private:
    TypeVector _argumentTypes;

private:
    bool _passArgumentsAsTargetValues;

};

} // namespace ir
} // namespace lucius








