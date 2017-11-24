/*  \file   Shape.cpp
    \author Gregory Diamos
    \date   October 1, 2017
    \brief  The source file for the Shape class.
*/

// Lucius Includes
#include <lucius/ir/interface/Shape.h>

#include <lucius/matrix/interface/Dimension.h>

namespace lucius
{
namespace ir
{

constexpr size_t UnknownValue = std::numeric_limits<size_t>::max();

class ShapeImplementation
{
public:
    ShapeImplementation(const matrix::Dimension& d)
    : _dimensions(d)
    {

    }

public:
    size_t size() const
    {
        return _dimensions.size();
    }

public:
    size_t& get(size_t i)
    {
        return _dimensions[i];
    }

    size_t get(size_t i) const
    {
        return _dimensions[i];
    }

public:
    void setUnknown()
    {
        for(auto& i : _dimensions)
        {
            i = UnknownValue;
        }
    }

private:
    matrix::Dimension _dimensions;
};

Shape::Shape()
: Shape(matrix::Dimension({}))
{

}

Shape::Shape(std::initializer_list<size_t> d)
: Shape(matrix::Dimension(d))
{

}

Shape::Shape(const matrix::Dimension& d)
: _implementation(std::make_unique<ShapeImplementation>(d))
{

}

Shape::Shape(const Shape& s)
: _implementation(std::make_unique<ShapeImplementation>(*s._implementation))
{

}

Shape::~Shape()
{

}

Shape& Shape::operator=(const Shape& s)
{
    *_implementation = *s._implementation;

    return *this;
}

void Shape::setUnknown()
{
    _implementation->setUnknown();
}

size_t& Shape::operator[](size_t i)
{
    return _implementation->get(i);
}

size_t Shape::operator[](size_t i) const
{
    return _implementation->get(i);
}

size_t Shape::size() const
{
    return _implementation->size();
}

inline bool isUnknown(size_t data)
{
    return data == UnknownValue;
}

size_t Shape::elements() const
{
    size_t count = 1;

    for(size_t i = 0; i < size(); ++i)
    {
        size_t dimensionSize = (*this)[i];

        if(isUnknown(dimensionSize))
        {
            return UnknownValue;
        }

        count *= dimensionSize;
    }

    return count;
}


} // namespace ir
} // namespace lucius


