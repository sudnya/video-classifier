/*  \file   IRExecutionEngineFactory.h
    \author Gregory Diamos
    \date   May 4, 2017
    \brief  The header file for the IRExecutionEngineFactory class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// forward declarations
namespace lucius { namespace runtime { class IRExecutionEngine; } }

namespace lucius { namespace ir { class Program; } }

namespace lucius
{
namespace runtime
{

/*! \brief A class used to construct IR execution engines. */
class IRExecutionEngineFactory
{
public:
    static std::unique_ptr<IRExecutionEngine> create(ir::Program& program);
};

} // namespace optimization
} // namespace lucius







