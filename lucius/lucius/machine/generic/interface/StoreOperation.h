/*  \file   StoreOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the StoreOperation class.
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

/*! \brief A class for representing a store operation. */
class StoreOperation : public ir::TargetOperation
{
public:
    StoreOperation();
    ~StoreOperation();

};

} // namespace generic
} // namespace machine
} // namespace lucius





