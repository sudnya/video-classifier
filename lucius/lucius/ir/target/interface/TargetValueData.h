/*  \file   TargetValueData.h
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The header file for the TargetValueData class.
*/

#pragma once

// Standard Library Includes
#include <memory>

// Forward Declarations
namespace lucius { namespace ir { class TargetValueDataImplementation; } }
namespace lucius { namespace ir { class TensorData;                    } }
namespace lucius { namespace ir { class IntegerData;                   } }

namespace lucius
{

namespace ir
{

/*! \brief A class for representing a value resource. */
class TargetValueData
{
public:
    TargetValueData(std::shared_ptr<TargetValueDataImplementation>);
    TargetValueData(TensorData);
    TargetValueData(IntegerData);
    TargetValueData();

public:
    void* data() const;

public:
    std::shared_ptr<TargetValueDataImplementation> getImplementation() const;

private:
    std::shared_ptr<TargetValueDataImplementation> _implementation;
};

template<typename To, typename From>
To data_cast(const From& from)
{
    return To(from.getImplementation());
}

} // namespace ir
} // namespace lucius







