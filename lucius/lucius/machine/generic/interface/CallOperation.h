/*! \file  CallOperation.h
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for the CallOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/target/interface/TargetOperation.h>

namespace lucius
{
namespace machine
{
namespace generic
{

class CallOperation : public ir::TargetOperation
{
public:
    CallOperation();
    ~CallOperation();
};

} // namespace generic
} // namespace machine
} // namespace lucius





