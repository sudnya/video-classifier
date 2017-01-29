
#pragma once

// Lucius Includes
#include <lucius/parallel/interface/cuda.h>
#include <lucius/parallel/interface/Memory.h>
#include <lucius/parallel/interface/String.h>
#include <lucius/parallel/interface/StringStream.h>
#include <lucius/parallel/interface/Set.h>

// Standard Library Includes
#include <cstring>
#include <string>
#include <sstream>

// Preprocessor Defines
#ifdef __NVCC__
#define ENABLE_LOGGING 1
#else
#define ENABLE_LOGGING 1
#endif

namespace lucius
{

namespace parallel
{

#if ENABLE_LOGGING
class LogDatabase
{
public:
    CUDA_DECORATOR LogDatabase()
    : _enableAllLogs(false)
    {

    }

public:
    CUDA_DECORATOR bool isLogEnabled(const string& logName) const
    {
        if(_enableAllLogs)
        {
            return true;
        }

        return _logsEnabled.count(logName) != 0;
    }

    CUDA_DECORATOR void enableSpecificLog(const string& logName)
    {
        _logsEnabled.insert(logName);
    }

    CUDA_DECORATOR void enableAllLogs(bool shouldAllLogsBeEnabled)
    {
        _enableAllLogs = shouldAllLogsBeEnabled;
    }

private:
    set<string> _logsEnabled;
    bool        _enableAllLogs;
};


#if defined(__NVCC__)
CUDA_GLOBAL_DECORATOR void allocateLogDatabase();

CUDA_GLOBAL_DECORATOR void enableSpecificLogDatabaseLog(const char* logName);
#endif

extern LogDatabase* hostLogDatabase;

#if defined(__NVCC__)
CUDA_MANAGED_DECORATOR LogDatabase* deviceLogDatabase = nullptr;
#endif

CUDA_DECORATOR inline LogDatabase* createAndGetLogDatabase()
{
    #if defined(__CUDA_ARCH__)
    return deviceLogDatabase;
    #else
    if(hostLogDatabase != nullptr)
    {
        return hostLogDatabase;
    }

    hostLogDatabase = new LogDatabase;
    #endif

    #if defined(__NVCC__)
    #if !defined(__CUDA_ARCH__)
    deviceLogDatabase = hostLogDatabase;
    #endif
    #endif

    return hostLogDatabase;
}

CUDA_DECORATOR inline void enableAllLogs(bool shouldAllLogsBeEnabled)
{
    createAndGetLogDatabase()->enableAllLogs(shouldAllLogsBeEnabled);
}

CUDA_DECORATOR inline void enableSpecificLog(const string& name)
{
    #if defined(__CUDA_ARCH__) || !defined(__NVCC__)
    createAndGetLogDatabase()->enableSpecificLog(name);
    #else
    createAndGetLogDatabase();

    char* data = reinterpret_cast<char*>(parallel::malloc(name.size() + 1));

    std::memcpy(data, name.c_str(), name.size() + 1);

    enableSpecificLogDatabaseLog<<<1, 1>>>(data);

    parallel::free(data);
    #endif
}

CUDA_DECORATOR inline bool isLogEnabled(const string& name)
{
    return createAndGetLogDatabase()->isLogEnabled(name);
}

class LogStream
{
public:
    CUDA_DECORATOR inline LogStream(const string& name)
    : _message("(" + name + ") : "), _isEnabled(isLogEnabled(name))
    {

    }

    CUDA_DECORATOR inline ~LogStream()
    {
        if(_isEnabled)
        {
            std::printf("%s", _message.c_str());
        }
    }

public:
    template <typename T>
    CUDA_DECORATOR LogStream& operator<<(T&& anything)
    {
        stringstream stream;

        stream << _message << anything;

        _message = stream.str();

        return *this;
    }

private:
    string _message;
    bool   _isEnabled;

};
#else

class LogStream
{
public:
    CUDA_DECORATOR LogStream(const string& name)
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

inline CUDA_DECORATOR LogStream log(const string& name)
{
    return LogStream(name);
}

} // namespace parallel

} // namespace lucius


