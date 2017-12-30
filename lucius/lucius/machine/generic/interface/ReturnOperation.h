/*! \file  ReturnOperation.h
    \date  December 24, 2017
    \autor Gregory Diamos <solusstultus@gmail.com>
    \brief The header file for the ReturnOperation class.
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

class ReturnOperation : public ir::TargetOperation
{
public:
    ReturnOperation();
    ~ReturnOperation();
};

} // namespace generic
} // namespace machine
} // namespace lucius






