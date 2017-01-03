/*  \file   TestEngine.h
    \date   October 28th, 2016
    \author Gregory Diamos <solusstultus@gmail.com>
    \brief  The header file for the TestEngine class.
*/

#pragma once

#include <functional>
#include <memory>
#include <string>

namespace lucius
{

namespace util
{

class TestEngineImplementation;

/*! \brief A class for managing unit tests. */
class TestEngine
{
public:
    typedef std::function<bool(void)> TestFunction;

public:
    TestEngine();
    ~TestEngine();

public:
    void addTest(const std::string& name, TestFunction);

public:
    std::string listTests() const;

public:
    bool run(const std::string& filter);

private:
    std::unique_ptr<TestEngineImplementation> _implementation;

};

}

}







