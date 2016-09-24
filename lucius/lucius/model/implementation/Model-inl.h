/*    \file   Model-inl.h
    \date   Saturday August 10, 2013
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The implementation file for the Model class.
*/

#pragma once

// Lucius Includes
#include <lucius/model/interface/Model.h>

#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <sstream>

namespace lucius
{

namespace model
{

template<typename T>
void Model::setAttribute(const std::string& name, const T& value)
{
    std::stringstream stream;

    stream << value;

    _attributes[name] = stream.str();
}

template<typename T>
T Model::getAttribute(const std::string& name) const
{
    auto attribute = _attributes.find(name);

    assertM(attribute != _attributes.end(), "Model is missing attribute '" + name + "'");

    T result;

    std::stringstream stream;

    stream << attribute->second;

    stream >> result;

    return result;
}

}

}

