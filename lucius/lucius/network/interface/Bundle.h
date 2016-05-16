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
    util::Any& operator[](const std::string& key);
    const util::Any& operator[](const std::string& key) const;

public:
    size_t size() const;

private:
    typedef std::map<std::string, util::Any> BundleData;

private:
    BundleData _data;

};

}
}


