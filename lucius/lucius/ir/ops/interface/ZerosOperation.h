/*  \file   ZerosOperation.h
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The header file for the ZerosOperation class.
*/

#pragma once

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

namespace lucius
{

namespace ir
{

/*! \brief A class for representing an operation that creates zeros. */
class ZerosOperation : public Operation
{
public:
    ZerosOperation(Type tensorType);
    ~ZerosOperation();

};

} // namespace ir
} // namespace lucius



