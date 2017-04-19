/*  \file   Loader.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Loader class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for loading the IR from a stream. */
class Loader
{
public:
    Loader(Context& context);

public:
    Program load(std::istream& stream);

};

} // namespace network
} // namespace lucius



