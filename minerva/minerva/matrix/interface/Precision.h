#pragma once

// Standard Library Includes
#include <tuple>

namespace minerva
{
namespace matrix
{

class Precision
{
public:
	enum Type
	{
		Half,
		Float,
		Double,
	};

public:
	Precision(Type t);

public:
	Type type() const;

public:
	size_t size() const;

private:
	Type _type;

};

class HalfPrecision : public Precision
{
public:
	HalfPrecision();

};

class SinglePrecision : public Precision
{
public:
	SinglePrecision();

};

class DoublePrecision : public Precision
{
public:
	DoublePrecision();

};

typedef std::tuple<HalfPrecision, SinglePrecision, DoublePrecision> AllPrecisions;

}
}

