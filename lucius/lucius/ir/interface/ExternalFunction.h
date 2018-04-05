/*  \file   ExternalFunction.h
    \author Gregory Diamos
    \date   March 17, 2018
    \brief  The header file for the ExternalFunction class.
*/

#pragma once

// Standard Library Includes
#include <memory>
#include <vector>

// Forward Declarations
namespace lucius { namespace ir { class ExternalFunctionImplementation; } }

namespace lucius { namespace ir { class ValueImplementation; } }
namespace lucius { namespace ir { class Type;                } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an external function exposed through the C ABI. */
class ExternalFunction
{
public:
    using TypeVector = std::vector<Type>;

public:
    ExternalFunction(const std::string& name, const TypeVector& argumentTypes);
    ExternalFunction(std::shared_ptr<ValueImplementation> implementation);
    ~ExternalFunction();

public:
    const std::string& name() const;

public:
    std::string toString() const;

public:
    Type getReturnType() const;

public:
    /*! \brief Should pointers to TargetValues be passed directly to the external function?

        Alternatively, the TargetValues are converted to the closest C type.
    */
    void setPassArgumentsAsTargetValues(bool condition);

    bool getPassArgumentsAsTargetValues() const;

public:
    std::shared_ptr<ExternalFunctionImplementation> getImplementation() const;
    std::shared_ptr<ValueImplementation> getValueImplementation() const;

private:
    std::shared_ptr<ExternalFunctionImplementation> _implementation;
};

} // namespace ir
} // namespace lucius






