/*! \file  LessOperation.h
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
namespace cpu
{

class LessOperation : public ir::TargetOperation
{
public:
    LessOperation();
    ~LessOperation();
};

} // namespace cpu
} // namespace machine
} // namespace lucius







