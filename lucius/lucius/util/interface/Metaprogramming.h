
#pragma once

// Standard Library Includes
#include <tuple>

namespace lucius
{
namespace util
{

template <typename T>
class RemoveFirstType
{
public:
    typedef std::tuple<> type;
};

template <typename T, typename... Ts>
class RemoveFirstType<std::tuple<T, Ts...>>
{
public:
    typedef std::tuple<Ts...> type;
};

template <size_t N, typename T>
class FillTuple
{
public:
    using NewType = std::tuple<T>;
    using RecursiveType = typename FillTuple<N-1, T>::type;
    using type = decltype(std::tuple_cat(NewType(), RecursiveType()));
};

template <typename T>
class FillTuple<0, T>
{
public:
    using type = std::tuple<>;

};

}
}




