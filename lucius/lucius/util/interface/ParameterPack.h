
#pragma once

// Standard Library Includes
#include <tuple>
#include <map>
#include <string>
#include <cassert>
#include <sstream>
#include <stdexcept>

namespace lucius
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
    bool contains(const std::string& name) const
    {
        return _parameters.count(name) != 0;
    }

public:
    template <typename T>
    T get(const std::string& name) const
    {
        auto parameter = _parameters.find(name);

        if(parameter == _parameters.end())
        {
            throw std::runtime_error("Parameter with name '" + name +
                "' not found in ParameterPack(" + toString() + ")");
        }

        T value;

        std::stringstream stream;

        stream << parameter->second;

        stream >> value;

        return value;
    }

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

public:
    template <typename T>
    void insert(const std::string& name, const T& value)
    {
        _parameters.insert(std::make_pair(name, _toString(value)));
    }

public:
    std::string toString() const
    {
        std::stringstream stream;

        bool first = true;

        for(auto& parameter : _parameters)
        {
            if(!first) stream << ", ";
            first = false;

            stream << parameter.first << " : " << parameter.second;
        }

        return stream.str();
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
        _parameters[std::get<0>(t)] = _toString(std::get<1>(t));
    }

    template<typename T>
    void _fill(const std::tuple<const char*, T>& t)
    {
        _parameters[std::get<0>(t)] = _toString(std::get<1>(t));
    }

private:
    template<typename T>
    std::string _toString(const T& t)
    {
        std::stringstream stream;

        stream << t;

        return stream.str();
    }

private:
    std::map<std::string, std::string> _parameters;

};

}

}





