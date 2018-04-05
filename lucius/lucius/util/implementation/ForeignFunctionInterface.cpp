/*! \file ForeignFunctionInterface.cpp
    \date March 15, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief The source file for hacked code required to assist windows
        compilaiton
*/

// Lucius Includes
#include <lucius/util/interface/ForeignFunctionInterface.h>

#include <lucius/util/interface/Metaprogramming.h>

// Standard Library Includes
#include <functional>
#include <map>

namespace lucius
{

namespace util
{

ForeignFunctionArgument::ForeignFunctionArgument(Type type)
: _sizet(0), _type(type)
{

}

ForeignFunctionArgument::ForeignFunctionArgument(size_t value)
: _sizet(value), _type(SizeT)
{

}

ForeignFunctionArgument::ForeignFunctionArgument(float value)
: _float(value), _type(Float)
{

}

ForeignFunctionArgument::ForeignFunctionArgument(void* pointer)
: _pointer(pointer), _type(Pointer)
{

}

ForeignFunctionArgument::Type ForeignFunctionArgument::getType() const
{
    return _type;
}

size_t ForeignFunctionArgument::getSizeT() const
{
    return _sizet;
}

float ForeignFunctionArgument::getFloat() const
{
    return _float;
}

void* ForeignFunctionArgument::getPointer() const
{
    return _pointer;
}

template<>
size_t ForeignFunctionArgument::get<size_t>() const
{
    return getSizeT();
}

template<>
float ForeignFunctionArgument::get<float>() const
{
    return getFloat();
}

template<>
void* ForeignFunctionArgument::get<void*>() const
{
    return getPointer();
}

ForeignFunctionSizeTArgument::ForeignFunctionSizeTArgument()
: ForeignFunctionArgument(SizeT)
{

}

ForeignFunctionFloatArgument::ForeignFunctionFloatArgument()
: ForeignFunctionArgument(Float)
{

}

ForeignFunctionPointerArgument::ForeignFunctionPointerArgument()
: ForeignFunctionArgument(Pointer)
{

}

class ForeignFunction
{
public:
    ForeignFunction(const std::string& name, void* functionPointer,
        const ForeignFunctionArguments& arguments)
    : _name(name), _functionPointer(functionPointer), _arguments(arguments)
    {

    }

public:
    const std::string& getName() const
    {
        return _name;
    }

    const ForeignFunctionArguments& getArguments() const
    {
        return _arguments;
    }

    void* getFunctionPointer() const
    {
        return _functionPointer;
    }

private:
    std::string _name;
    void* _functionPointer;
    ForeignFunctionArguments _arguments;
};

template<typename... Args>
class GetFunctionSignatureFromTuple
{
public:
    using type = void(void);
};

template<typename... Args>
class GetFunctionSignatureFromTuple<std::tuple<Args...>>
{
public:
    using type = void(Args...);
};

template<typename ArgumentType, typename ArgumentList>
void dispatchFunction(void* functionPointer, const ForeignFunctionArguments& dynamicArguments,
    size_t dynamicIndex, const ArgumentList& staticArguments,
    const std::tuple<std::tuple<ArgumentType>>& space)
{
    assert(dynamicIndex + 1 == dynamicArguments.size());
    assert(ArgumentType().getType() == dynamicArguments.back().getType());

    using NativeType = typename ArgumentType::NativeType;

    auto staticArgument = std::tuple<NativeType>(dynamicArguments.back().get<NativeType>());
    auto newStaticArguments = std::tuple_cat(staticArguments, staticArgument);

    using NewArgumentList = decltype(newStaticArguments);

    using FunctionType = typename GetFunctionSignatureFromTuple<NewArgumentList>::type;

    auto function = std::function<FunctionType>(reinterpret_cast<FunctionType*>(functionPointer));

    std::apply(function, newStaticArguments);
}

template<typename ArgumentTypeSpace, typename ArgumentList>
void dispatchFunction(void* functionPointer, const ForeignFunctionArguments& dynamicArguments,
    size_t dynamicIndex, const ArgumentList& staticArguments,
    const std::tuple<ArgumentTypeSpace>& space)
{
    using PossibleArgumentType = typename std::tuple_element<0, ArgumentTypeSpace>::type;

    assert(dynamicIndex < dynamicArguments.size());
    auto dynamicArgument = dynamicArguments[dynamicIndex];

    if(PossibleArgumentType().getType() == dynamicArgument.getType())
    {
        dispatchFunction(functionPointer, dynamicArguments, dynamicIndex, staticArguments,
            std::tuple<std::tuple<PossibleArgumentType>>());
    }
    else
    {
        using NewArgumentTypeSpace = typename RemoveFirstType<ArgumentTypeSpace>::type;

        dispatchFunction(functionPointer, dynamicArguments, dynamicIndex, staticArguments,
            std::tuple<NewArgumentTypeSpace>());
    }
}

template<typename... ArgumentTypeSpace, typename ArgumentList>
void dispatchFunction(void* functionPointer, const ForeignFunctionArguments& dynamicArguments,
    size_t dynamicIndex,
    const ArgumentList& staticArguments, const std::tuple<std::tuple<>, ArgumentTypeSpace...>& space)
{
    dispatchFunction(functionPointer, dynamicArguments, dynamicIndex, staticArguments,
        std::tuple<ArgumentTypeSpace...>());
}

template<typename SearchSpace, typename ArgumentList>
void dispatchFunction(void* functionPointer, const ForeignFunctionArguments& dynamicArguments,
    size_t dynamicIndex, const ArgumentList& staticArguments, const SearchSpace& space)
{
    using ArgumentTypeSpace = typename std::tuple_element<0, SearchSpace>::type;
    using PossibleArgumentType = typename std::tuple_element<0, ArgumentTypeSpace>::type;

    assert(dynamicIndex < dynamicArguments.size());
    auto dynamicArgument = dynamicArguments[dynamicIndex];

    if(PossibleArgumentType().getType() == dynamicArgument.getType())
    {
        using NewSearchSpace = typename RemoveFirstType<SearchSpace>::type;
        using NativeType = typename PossibleArgumentType::NativeType;

        auto staticArgument = std::tuple<NativeType>(dynamicArgument.get<NativeType>());

        auto newStaticArguments = std::tuple_cat(staticArguments, staticArgument);
        auto newSpace = NewSearchSpace();

        dispatchFunction(functionPointer, dynamicArguments, dynamicIndex + 1, newStaticArguments,
            newSpace);
    }
    else
    {
        using TempSearchSpace = typename RemoveFirstType<SearchSpace>::type;
        using NewArgumentTypeSpace = typename RemoveFirstType<ArgumentTypeSpace>::type;

        using NewSearchSpace = decltype(std::tuple_cat(std::tuple<NewArgumentTypeSpace>(),
            TempSearchSpace()));

        dispatchFunction(functionPointer, dynamicArguments, dynamicIndex, staticArguments,
            NewSearchSpace());
    }
}

template<int argumentCount>
class GetArgumentSearchSpace
{
public:
    using ArgumentSpace = std::tuple<ForeignFunctionSizeTArgument,
                                     ForeignFunctionFloatArgument,
                                     ForeignFunctionPointerArgument>;
    using type = typename FillTuple<argumentCount, ArgumentSpace>::type;

};

template<int argumentCount>
void dispatchFunction(void* functionPointer, const ForeignFunctionArguments& arguments)
{
    using SearchSpace = typename GetArgumentSearchSpace<argumentCount>::type;

    dispatchFunction(functionPointer, arguments, 0, std::tuple<>(), SearchSpace());
}

template<int maximumArguments>
void dispatch(void* functionPointer, const ForeignFunctionArguments& arguments)
{
    if(arguments.size() == maximumArguments)
    {
        dispatchFunction<maximumArguments>(functionPointer, arguments);
    }
    else
    {
        constexpr int argumentCount = maximumArguments > 1 ? maximumArguments - 1 : 1;

        dispatch<argumentCount>(functionPointer, arguments);
    }
}

static std::map<std::string, ForeignFunction> foreignFunctions;

void registerForeignFunction(const std::string& name, void* function,
    const ForeignFunctionArguments& arguments)
{
    foreignFunctions.insert(std::make_pair(name, ForeignFunction(name, function, arguments)));
}

bool isForeignFunctionRegistered(const std::string& name)
{
    return foreignFunctions.count(name) != 0;
}

void checkArguments(const ForeignFunctionArguments& left, const ForeignFunctionArguments& right)
{
    if(left.size() != right.size())
    {
        throw std::runtime_error("Foreign function call does not match signature.");
    }

    for(auto l = left.begin(), r = right.begin(); l != left.end(); ++l, ++r)
    {
        if(l->getType() != r->getType())
        {
            throw std::runtime_error("Foreign function call does not match signature.");
        }
    }
}

void callForeignFunction(const std::string& name, const ForeignFunctionArguments& arguments)
{
    auto function = foreignFunctions.find(name);

    assert(function != foreignFunctions.end());

    checkArguments(function->second.getArguments(), arguments);

    if(arguments.empty())
    {
        using VoidFunctionType = void(void);

        std::function<VoidFunctionType>(reinterpret_cast<VoidFunctionType*>(
            function->second.getFunctionPointer()));
    }
    else
    {
        dispatch<3>(function->second.getFunctionPointer(), arguments);
    }
}

}

}



