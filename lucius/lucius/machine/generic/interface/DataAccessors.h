/*  \file   DataAccessors.h
    \author Gregory Diamos
    \date   April 24, 2018
    \brief  The header file for the DataAccessors functions.
*/

#pragma once

// Forward Declarations
namespace lucius { namespace ir { class TargetValueData; } }
namespace lucius { namespace ir { class Use;             } }
namespace lucius { namespace ir { class Type;            } }

namespace lucius { namespace matrix { class Matrix;    } }
namespace lucius { namespace matrix { class Operator;  } }
namespace lucius { namespace matrix { class Dimension; } }

// Standard Library Includes
#include <cstddef>

namespace lucius
{
namespace machine
{
namespace generic
{

size_t getDataAsInteger(const ir::Use& );

float getDataAsFloat(const ir::Use& );

void* getDataAsPointer(const ir::Use& );

matrix::Operator getDataAsOperator(const ir::Use& );

matrix::Matrix getDataAsTensor(const ir::Use& );

matrix::Dimension getDataAsDimension(const ir::Use& );

/*! In the case of a compound type (e.g. structure), get the data of the index'th element */
ir::TargetValueData getDataAtIndex(const ir::Use& value, size_t index);

void copyData(ir::TargetValueData destination, ir::TargetValueData source, const ir::Type& type);

} // namespace generic
} // namespace machine
} // namespace lucius



