/*  \file   Operation.cpp
    \author Gregory Diamos
    \date   April 4, 2017
    \brief  The source file for the Operation class.
*/

namespace lucius
{

namespace ir
{

Operation::Operation(const std::string& name,
    const ArgumentList& inputs, const ArgumentList& outputs)
: _name(name), _inputs(inputs), _outputs(outputs)
{

}

Operation::~Operation()
{
    // intentionally blank
}

} // namespace ir
} // namespace lucius




