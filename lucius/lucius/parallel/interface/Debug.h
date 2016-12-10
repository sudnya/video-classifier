
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>

// Standard Library Includes
#include <string>
#include <sstream>

namespace lucius
{

namespace parallel
{

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
    LogStream& operator<<(T&& anything)
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

inline CUDA_DECORATOR LogStream log(const std::string& name)
{
    return LogStream(name);
}

} // namespace parallel

} // namespace lucius


