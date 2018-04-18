/*! \file  BranchOperation.h
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for the BranchOperation class.
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

class BranchOperation : public ir::TargetOperation
{
public:
    BranchOperation();
    ~BranchOperation();
};

} // namespace generic
} // namespace machine
} // namespace lucius







