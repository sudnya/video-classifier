/*  \file   Bundle.cpp
    \author Gregory Diamos
    \date   May 15, 2016
    \brief  The source file for the Bundle class.
*/

// Lucius Includes
#include <lucius/network/interface/Bundle.h>

// Standard Library Includes
#include <map>
#include <string>
#include <cassert>

namespace lucius
{
namespace network
{

Bundle::Bundle()
{

}

util::Any& Bundle::operator[](const std::string& key)
{
    return _data[key];
}

const util::Any& Bundle::operator[](const std::string& key) const
{
    auto value = _data.find(key);

    assert(value != _data.end());

    return value->second;
}

size_t Bundle::size() const
{
    return _data.size();
}

}
}



