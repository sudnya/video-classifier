/*  \file   LoadOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the LoadOperation class.
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
class LoadOperation : public ir::TargetOperation
{
public:
    LoadOperation();
    ~LoadOperation();

};

} // namespace generic
} // namespace machine
} // namespace lucius






