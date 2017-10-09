/*  \file   Operation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Operation class.
*/

// Lucius Includes
#include <lucius/ir/interface/Operation.h>

#include <lucius/ir/interface/Use.h>

// Forward Declarations
namespace lucius { namespace ir { class BasicBlockImplementation; } }

namespace lucius
{

namespace ir
{

class OperationImplementation : public User
{
private:
    std::weak_ptr<BasicBlockImplementation> _parent;

};

Operation::Operation()
: _implementation(std::make_shared<OperationImplementation>())
{

}

Operation::~Operation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius

