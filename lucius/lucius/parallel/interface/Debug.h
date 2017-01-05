
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/String.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace parallel
{

#if ENABLE_LOGGING
CUDA_DECORATOR bool isLogEnabled(const std::string& name);

class LogStream
{
public:
    CUDA_DECORATOR LogStream(const std::string& name)
    : _message("(" + name + ") : "), _isEnabled(isLogEnabled(name))
    {

    }

    CUDA_DECORATOR ~LogStream();

public:
    template <typename T>
    CUDA_DECORATOR LogStream& operator<<(T&& anything)
    {
        std::stringstream stream;

        stream << _message << anything;

        _message = stream.str();

        return *this;
    }

private:
    std::string _message;
    bool        _isEnabled;

};
#else

class LogStream
{
public:
    CUDA_DECORATOR LogStream(const parallel::string& name)
    {

    }

    CUDA_DECORATOR ~LogStream()
    {

    }

public:
    template <typename T>
    CUDA_DECORATOR LogStream& operator<<(T&& anything)
    {
        return *this;
    }

};

#endif

inline CUDA_DECORATOR LogStream log(const parallel::string& name)
{
    return LogStream(name);
}

} // namespace parallel

} // namespace lucius


