/*! \file PointerCasts.h
    \date May 4, 2018
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for a set of non-standard pointer casts.
*/

#pragma once

// Standard Library Includes
#include <memory>

namespace lucius
{
namespace util
{

template <typename D, typename S>
std::unique_ptr<D> unique_pointer_cast(std::unique_ptr<S>&& s)
{
    return std::unique_ptr<D>(static_cast<D*>(s.release()));
}

} // namespace util
} // namespace lucius


