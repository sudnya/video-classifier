/*! \file  SRandOperation.h
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for the SRandOperation class.
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

class SRandOperation : public ir::TargetOperation
{
public:
    SRandOperation();
    ~SRandOperation();
};

} // namespace cpu
} // namespace machine
} // namespace lucius







