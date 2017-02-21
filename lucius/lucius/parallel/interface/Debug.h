
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
#define ENABLE_LOGGING 1

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
        #ifdef __NVCC__
        if(this == nullptr)
        {
            return false;
        }
        #endif

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

CUDA_DECORATOR LogDatabase* createAndGetLogDatabase();
LogDatabase* createAndGetHostLogDatabase();

void enableSpecificDeviceLog(const string& name);
void enableAllDeviceLogs(bool shouldAllLogsBeEnabled);

inline void enableAllLogs(bool shouldAllLogsBeEnabled)
{
    enableAllDeviceLogs(shouldAllLogsBeEnabled);
    createAndGetHostLogDatabase()->enableAllLogs(shouldAllLogsBeEnabled);
}

inline void enableSpecificLog(const string& name)
{
    createAndGetHostLogDatabase()->enableSpecificLog(name);
    enableSpecificDeviceLog(name);
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


