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
		NoType,
		Half,
		Single,
		Double,
	};

public:
	Precision();
	Precision(Type t);

public:
	Type type() const;

public:
	size_t size() const;

public:
	bool operator==(const Precision&) const;

private:
	Type _type;

};

class HalfPrecision : public Precision
{
public:
	typedef float type;

public:
	HalfPrecision();

};

class SinglePrecision : public Precision
{
public:
	typedef float type;

public:
	SinglePrecision();

};

class DoublePrecision : public Precision
{
public:
	typedef double type;

public:
	DoublePrecision();

};

typedef std::tuple<HalfPrecision, SinglePrecision, DoublePrecision> AllPrecisions;

}
}

