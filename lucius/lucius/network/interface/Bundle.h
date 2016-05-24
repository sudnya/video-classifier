/*  \file   Bundle.h
    \author Gregory Diamos
    \date   May 15, 2016
    \brief  The interface for the Bundle class.
*/

#pragma once

// Lucius Includes
#include <lucius/util/interface/Any.h>

// Standard Library Includes
#include <map>
#include <string>

namespace lucius
{
namespace network
{

/*! \brief A container for generic untyped data. */
class Bundle
{
public:
    Bundle();

    template <typename... Args>
    Bundle(Args... args)
    {
        _fill(args...);
    }

public:
    util::Any& operator[](const std::string& key);
    const util::Any& operator[](const std::string& key) const;

public:
    size_t size() const;

private:
    template <typename T, typename U>
    void _fill(std::pair<T, U> argument)
    {
        _data[argument.first] = argument.second;
    }

    template <typename T, typename U, typename... Args>
    void _fill(std::pair<T, U> argument, Args... args)
    {
        _fill(argument);
        _fill(args...);
    }

private:
    typedef std::map<std::string, util::Any> BundleData;

private:
    BundleData _data;

};

}
}


