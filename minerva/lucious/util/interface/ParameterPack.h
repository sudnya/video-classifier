
#pragma once

// Standard Library Includes
#include <tuple>
#include <map>
#include <string>
#include <cassert>
#include <sstream>

namespace lucious
{

namespace util
{

class ParameterPack
{
public:
    template <typename... Args>
    ParameterPack(Args... args)
    {
        _fill(args...);
    }

    ParameterPack()
    {

    }

public:
    template <typename T>
    T get(const std::string& name, const T& defaultValue) const
    {
        auto parameter = _parameters.find(name);

        if(parameter == _parameters.end())
        {
            return defaultValue;
        }

        T value;

        std::stringstream stream;

        stream << parameter->second;

        stream >> value;

        return value;
    }

private:
    template <typename T, typename... Args>
    void _fill(std::tuple<std::string, T> t, Args... args)
    {
        _fill(t);
        _fill(args...);
    }

    template <typename T, typename... Args>
    void _fill(std::tuple<const char*, T> t, Args... args)
    {
        _fill(t);
        _fill(args...);
    }

    template<typename T>
    void _fill(const std::tuple<std::string, T>& t)
    {
        _parameters[std::get<0>(t)] = std::to_string(std::get<1>(t));
    }

    template<typename T>
    void _fill(const std::tuple<const char*, T>& t)
    {
        _parameters[std::get<0>(t)] = std::to_string(std::get<1>(t));
    }

private:
    std::map<std::string, std::string> _parameters;

};

}

}





