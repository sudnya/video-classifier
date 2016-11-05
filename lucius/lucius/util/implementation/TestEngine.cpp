/*  \file   TestEngine.cpp
    \date   October 28th, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The source file for the TestEngine class.
*/

// Lucius Includes
#include <lucius/util/interface/TestEngine.h>
#include <lucius/util/interface/memory.h>

// Standard Library Includes
#include <map>
#include <regex>

namespace lucius
{

namespace util
{

class TestEngineImplementation
{
public:
    bool run(const std::string& filter)
    {
        bool result = true;

        for(auto& test : _tests)
        {
            if(!_nameMatchesFilter(test.first, filter))
            {
                continue;
            }

            result &= test.second();
        }

        return result;
    }

    void addTest(const std::string& name, TestEngine::TestFunction function)
    {
        _tests[name] = function;
    }

    std::string listTests() const
    {
        std::string results;

        for(auto& test : _tests)
        {
            results += test.first + "\n";
        }

        return results;
    }

private:
    bool _nameMatchesFilter(const std::string& name, const std::string& filter)
    {
        if(filter.empty())
        {
            return true;
        }

        std::regex regularExpression(filter);
        std::smatch matches;

        return std::regex_match(name, matches, regularExpression);
    }

private:
    std::map<std::string, TestEngine::TestFunction> _tests;

};

TestEngine::TestEngine()
: _implementation(std::make_unique<TestEngineImplementation>())
{

}

TestEngine::~TestEngine()
{
    // Intentionally Blank
}

void TestEngine::addTest(const std::string& name, TestFunction function)
{
    _implementation->addTest(name, function);
}

std::string TestEngine::listTests() const
{
    return _implementation->listTests();
}

bool TestEngine::run(const std::string& filter)
{
    return _implementation->run(filter);
}

}

}








