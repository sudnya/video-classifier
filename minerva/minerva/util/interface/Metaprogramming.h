
#pragma once

// Standard Library Includes
#include <tuple>

namespace lucious
{
namespace util
{

template<typename T>
struct RemoveFirstType
{
	typedef std::tuple<> type;
};

template<typename T, typename... Ts>
struct RemoveFirstType<std::tuple<T, Ts...>>
{
    typedef std::tuple<Ts...> type;
};

}
}




