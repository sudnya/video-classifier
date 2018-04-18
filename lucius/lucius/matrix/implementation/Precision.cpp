
// Lucius Includes
#include <lucius/matrix/interface/Precision.h>

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/Knobs.h>

namespace lucius
{
namespace matrix
{

Precision::Precision()
: _type(NoType)
{

}

Precision::Precision(Type t)
: _type(t)
{

}

Precision::Type Precision::type() const
{
    return _type;
}

size_t Precision::size() const
{
    switch(type())
    {
    case Half: return sizeof(float)/2;
    case Single: return sizeof(float);
    case Double: return sizeof(double);
    case SizeT: return sizeof(size_t);
    default:
        assertM(false, "Invalid type.");
        return 0;
    }
}

bool Precision::operator==(const Precision& p) const
{
    return type() == p.type();
}

bool Precision::operator!=(const Precision& p) const
{
    return !(*this == p);
}

Precision Precision::getDefaultPrecision()
{
    auto precision = util::KnobDatabase::getKnobValue(
        "Matrix::DefaultPrecision", "SinglePrecision");

    if(precision == "SinglePrecision")
    {
        return SinglePrecision();
    }
    else if(precision == "HalfPrecision")
    {
        return HalfPrecision();
    }
    else if(precision == "SizeTPrecision")
    {
        return SizeTPrecision();
    }

    return DoublePrecision();
}


std::string Precision::toString() const
{
    switch(type())
    {
    case Half: return "HalfPrecision";
    case Single: return "SinglePrecision";
    case Double: return "DoublePrecision";
    case SizeT: return "SizeTPrecision";
    default:
        return "Invalid";
    }
}

std::unique_ptr<Precision> Precision::fromString(const std::string& name)
{
    if(name == "HalfPrecision")
    {
        return std::unique_ptr<Precision>(new HalfPrecision());
    }
    else if(name == "SinglePrecision")
    {
        return std::unique_ptr<Precision>(new SinglePrecision());
    }
    else if(name == "DoublePrecision")
    {
        return std::unique_ptr<Precision>(new DoublePrecision());
    }
    else if(name == "SizeTPrecision")
    {
        return std::unique_ptr<Precision>(new DoublePrecision());
    }

    return std::unique_ptr<Precision>(nullptr);
}

HalfPrecision::HalfPrecision()
: Precision(Half)
{

}

SinglePrecision::SinglePrecision()
: Precision(Single)
{

}

DoublePrecision::DoublePrecision()
: Precision(Double)
{

}

SizeTPrecision::SizeTPrecision()
: Precision(SizeT)
{

}

}
}

