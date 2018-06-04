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
    bool empty() const
    {
        return _dimensions.empty();
    }

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

public:
    using       iterator = size_t*;
    using const_iterator = const size_t*;

public:
    iterator begin()
    {
        return _dimensions.begin();
    }

    const_iterator begin() const
    {
        return _dimensions.begin();
    }

    iterator end()
    {
        return _dimensions.end();
    }

    const_iterator end() const
    {
        return _dimensions.end();
    }

public:
    std::string toString() const
    {
        return _dimensions.toString();
    }

public:
    const matrix::Dimension& getDimension() const
    {
        return _dimensions;
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

bool Shape::areAllDimensionsKnown() const
{
    for(auto& i : *this)
    {
        if(i == UnknownValue)
        {
            return false;
        }
    }

    return true;
}

Shape::iterator Shape::begin()
{
    return _implementation->begin();
}

Shape::const_iterator Shape::begin() const
{
    return _implementation->begin();
}

Shape::iterator Shape::end()
{
    return _implementation->end();
}

Shape::const_iterator Shape::end() const
{
    return _implementation->end();
}

size_t& Shape::operator[](size_t i)
{
    return _implementation->get(i);
}

size_t Shape::operator[](size_t i) const
{
    return _implementation->get(i);
}

bool Shape::empty() const
{
    return _implementation->empty();
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

std::string Shape::toString() const
{
    return _implementation->toString();
}

matrix::Dimension Shape::getDimension() const
{
    assert(areAllDimensionsKnown());

    return _implementation->getDimension();
}

} // namespace ir
} // namespace lucius


