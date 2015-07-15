#pragma once

// Standard Library Includes
#include <tuple>

namespace lucius
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
    explicit Precision(Type t);

public:
    Type type() const;

public:
    size_t size() const;

public:
    bool operator==(const Precision&) const;
    bool operator!=(const Precision&) const;

public:
    static Precision getDefaultPrecision();

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

