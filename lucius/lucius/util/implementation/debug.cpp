
/*!    \file debug.cpp
*
*    \brief Source file for common debug macros
*
*    \author Gregory Diamos
*
*    \date : Wednesday April 29, 2009
*
*/

#include <lucius/util/interface/debug.h>
#include <lucius/util/interface/string.h>

#include <lucius/parallel/interface/Debug.h>

// Standard Library Includes
#include <memory>
#include <unordered_set>
#include <fstream>

namespace lucius
{

namespace util
{

/*! \brief Global report timer */
Timer _ReportTimer;

std::string _debugTime()
{
    std::stringstream stream;
    stream.setf( std::ios::fixed, std::ios::floatfield );
    stream.precision( 6 );
    stream << _ReportTimer.seconds();
    return stream.str();
}

std::string _debugFile( const std::string& file, unsigned int line )
{
    std::stringstream lineColon;
    lineColon << line << ":";

    std::stringstream stream;

    stream << stripReportPath<'/'>( file ) << ":";
    stream.width( 5 );
    stream.fill( ' ' );
    stream << std::left << lineColon.str();
    return stream.str();
}

/*! \brief Global logging infrastructure */
class LogDatabase
{
public:
    LogDatabase();

public:
    typedef std::unordered_set<std::string> StringSet;

public:
    bool enableAll;
    StringSet enabledLogs;


public:
    bool isEnabled(const std::string& logName) const
    {
        return enableAll || (enabledLogs.count(logName) != 0);
    }

public:
    std::ostream& getLog()
    {
        if(_logFile)
        {
            _logFile->flush();

            return *_logFile;
        }

        return std::cout;
    }

public:
    void setLogFile(const std::string& name)
    {
        _logFile.reset(new std::ofstream(name));

        if(!_logFile->is_open())
        {
            throw std::runtime_error("Failed to open log file '" + name + "' for writing.");
        }
    }

private:
    std::unique_ptr<std::ofstream> _logFile;
};

LogDatabase::LogDatabase()
: enableAll(false)
{

}

static LogDatabase logDatabase;

void enableAllLogs()
{
    logDatabase.enableAll = true;
    parallel::enableAllLogs(true);
}

void enableSpecificLogs(const std::string& modules)
{
    auto individualModules = util::split(modules, ",");

    for(auto& module : individualModules)
    {
        enableLog(module);
    }
}

void setLogFile(const std::string& name)
{
    logDatabase.setLogFile(name);
}

void enableLog(const std::string& name)
{
    logDatabase.enabledLogs.insert(name);
    parallel::enableSpecificLog(name.c_str());
}

static std::unique_ptr<NullStream> nullstream;

std::ostream& _getStream(const std::string& name)
{
    if(logDatabase.isEnabled(name))
    {
        logDatabase.getLog() << "(" << _debugTime() << "): " << name << ": ";

        return logDatabase.getLog();
    }

    if(nullstream == nullptr)
    {
        nullstream.reset(new NullStream);
    }

    return *nullstream;
}

bool isLogEnabled(const std::string& name)
{
    return logDatabase.isEnabled(name);
}

}

}

