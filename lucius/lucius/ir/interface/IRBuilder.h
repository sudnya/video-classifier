/*  \file   IRBuilder.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the IRBuilder class.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class IRBuilder; } }

namespace lucius
{

namespace ir
{

/*! \brief Helps build the IR. */
class IRBuilder
{
public:
    IRBuilder(Context& context);

public:
    /*! \brief Extracts the program from the builder, caller takes ownership. */
    std::unique_ptr<Program> getProgram();



};

} // namespace ir
} // namespace lucius






