#pragma once

// Minerva Includes
#include <minerva/matrix/interface/Precision.h>

namespace minerva
{
namespace matrix
{

class FloatReference
{
public:
    FloatReference(const Precision& p, void* d);

public:
    FloatReference& operator=(double);
    FloatReference& operator+=(double);
    FloatReference& operator-=(double);

public:
    operator double() const;

public:
          void* address();
    const void* address() const;

private:
    Precision _precision;
    void*     _data;

};

class ConstFloatReference
{
public:
    ConstFloatReference(const Precision& p, const void* d);

public:
    operator double() const;

public:
    const void* address() const;

private:
    Precision   _precision;
    const void* _data;
};

}
}




