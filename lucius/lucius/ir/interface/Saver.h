/*  \file   Saver.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the Saver class.
*/

#pragma once

namespace lucius
{

namespace ir
{

/*! \brief A class for saving the IR to a stream. */
class Saver
{
public:
    Saver(Context& context);

public:
    void save(std::ostream& stream, const Program& program);


};

} // namespace network
} // namespace lucius




