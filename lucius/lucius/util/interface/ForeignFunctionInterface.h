/*! \file ForeignFunctionInterface.h
    \date March 15, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for hacked code required to assist windows
        compilaiton
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace lucius
{

namespace util
{

class ForeignFunctionArgument
{
public:
    enum Type
    {
        SizeT,
        Float,
        Pointer
    };

public:
    ForeignFunctionArgument(Type type);
    explicit ForeignFunctionArgument(size_t value);
    explicit ForeignFunctionArgument(float value);
    explicit ForeignFunctionArgument(void* pointer);

public:
    Type getType() const;

public:
    size_t getSizeT() const;
    float getFloat() const;
    void* getPointer() const;

public:
    template<typename T>
    T get() const;

private:
    union
    {
        size_t _sizet;
        float  _float;
        void*  _pointer;
    };

    Type _type;
};

class ForeignFunctionSizeTArgument : public ForeignFunctionArgument
{
public:
    using NativeType = size_t;

public:
    ForeignFunctionSizeTArgument();
};

class ForeignFunctionFloatArgument : public ForeignFunctionArgument
{
public:
    using NativeType = float;

public:
    ForeignFunctionFloatArgument();
};

class ForeignFunctionPointerArgument : public ForeignFunctionArgument
{
public:
    using NativeType = void*;

public:
    ForeignFunctionPointerArgument();
};

using ForeignFunctionArguments = std::vector<ForeignFunctionArgument>;

void registerForeignFunction(const std::string& name, void* function,
    const ForeignFunctionArguments& arguments);
bool isForeignFunctionRegistered(const std::string& name);

void callForeignFunction(const std::string& name, const ForeignFunctionArguments& arguments);

}

}


