/*    \file   PropertyTree-inl.h
    \date   Saturday August 10, 2015
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The template implementation file for the PropertyTree class.
*/

#pragma once

// Lucius Includes
#include <lucius/util/interface/PropertyTree.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace util
{

namespace
{

template<typename T>
T fromString(const std::string& s)
{
    std::stringstream stream;

    stream << s;

    T result;

    stream >> result;

    return result;
}

template<typename T>
std::string toString(const T& t)
{
    std::stringstream stream;

    stream << t;

    return stream.str();
}

}

template<typename T>
PropertyTree& PropertyTree::operator=(const T& t)
{
    return operator=(toString(t));
}

template<typename T>
T PropertyTree::get(const std::string& field) const
{
    return fromString<T>(operator[](field));
}

template<typename T>
T PropertyTree::get(const std::string& field, const T& defaultValue) const
{
    if(!exists(field))
    {
        return defaultValue;
    }

    return get<T>(field);
}

template<typename T>
T PropertyTree::key() const
{
    return fromString<T>(key());
}

template<typename T>
PropertyTree& PropertyTree::operator[](const T& key)
{
    return operator[](toString(key));
}

template<typename T>
const PropertyTree& PropertyTree::operator[](const T& key) const
{
    return operator[](toString(key));
}

}

}

