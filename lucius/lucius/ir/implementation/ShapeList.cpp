/*  \file   ShapeList.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the ShapeList class.
*/

// Lucius Includes
#include <lucius/ir/interface/ShapeList.h>
#include <lucius/ir/interface/Shape.h>

namespace lucius
{
namespace ir
{

ShapeList::ShapeList()
{

}

ShapeList::ShapeList(std::initializer_list<Shape> s)
: _shapes(s)
{

}

ShapeList::~ShapeList()
{

}

Shape& ShapeList::front()
{
    return _shapes.front();
}

const Shape& ShapeList::front() const
{
    return _shapes.front();
}

} // namespace ir
} // namespace lucius



