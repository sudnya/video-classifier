
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/cstring.h>
#include <lucius/parallel/interface/ScalarOperations.h>

namespace lucius
{
namespace parallel
{

/*! \brief A CUDA compatible stringsteam class with C++ standard library syntax and semantics. */
class stringstream
{
public:
    CUDA_DECORATOR stringstream()
    {

    }

public:
    CUDA_DECORATOR string str() const
    {
        return _string;
    }

public:
    CUDA_DECORATOR stringstream& operator<<(bool val)
    {
        _string += to_string(static_cast<long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(short val)
    {
        _string += to_string(static_cast<long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(unsigned short val)
    {
        _string += to_string(static_cast<unsigned long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(int val)
    {
        _string += to_string(static_cast<long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(unsigned int val)
    {
        _string += to_string(static_cast<unsigned long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(long val)
    {
        _string += to_string(static_cast<long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(unsigned long val)
    {
        _string += to_string(static_cast<unsigned long long>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(long long val)
    {
        _string += to_string(val);

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(unsigned long long val)
    {
        _string += to_string(val);

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(float val)
    {
        _string += to_string(static_cast<double>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(double val)
    {
        _string += to_string(val);

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(void* val)
    {
        _string += to_hex(reinterpret_cast<long long int>(val));

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(const char* val)
    {
        _string += string(val);

        return *this;
    }

    CUDA_DECORATOR stringstream& operator<<(const string& val)
    {
        _string += val;

        return *this;
    }

private:
    string _string;

};

} // namespace parallel

} // namespace lucius

