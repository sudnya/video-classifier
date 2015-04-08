
// Minerva Includes
#include <minerva/matrix/interface/Precision.h>

#include <minerva/util/interface/debug.h>

namespace minerva
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
	default:
		assertM(false, "Invalid type.");
		return 0;
	}
}

bool Precision::operator==(const Precision& p) const
{
	return type() == p.type();
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

}
}

