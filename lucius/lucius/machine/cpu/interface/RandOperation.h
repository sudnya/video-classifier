/*! \file  RandOperation.h
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for the RandOperation class.
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

class RandOperation : public ir::TargetOperation
{
public:
    RandOperation();
    ~RandOperation();
};

} // namespace cpu
} // namespace machine
} // namespace lucius








