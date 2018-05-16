/*  \file   test-lazy-ir.cpp
    \date   April 17, 2017
    \author Gregory Diamos
    \brief  The source file for the lazy ir unit tests class.
*/

// Lucius Includes
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>

#include <lucius/lazy-ir/interface/LazyIr.h>
#include <lucius/lazy-ir/interface/LazyValue.h>
#include <lucius/lazy-ir/interface/MatrixOperations.h>
#include <lucius/lazy-ir/interface/CopyOperations.h>
#include <lucius/lazy-ir/interface/CastOperations.h>
#include <lucius/lazy-ir/interface/RandomOperations.h>
#include <lucius/lazy-ir/interface/GradientOperations.h>
#include <lucius/lazy-ir/interface/Initializers.h>
#include <lucius/lazy-ir/interface/Loops.h>
#include <lucius/lazy-ir/interface/Operators.h>

#include <lucius/util/interface/debug.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/TestEngine.h>

using Matrix = lucius::matrix::Matrix;
using LazyValue = lucius::lazy::LazyValue;
using SinglePrecision = lucius::matrix::SinglePrecision;

/*
    Test matrix addition

    [ 1 2 ] + [ 2 3 ] = [  3  5 ]
    [ 3 4 ]   [ 4 5 ]   [  7  9 ]
    [ 5 6 ]   [ 6 7 ]   [ 11 13 ]

*/
static bool testAdd()
{
    Matrix a(3, 2);
    Matrix b(3, 2);
    Matrix c(3, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;

    b(0, 0) = 2;
    b(0, 1) = 3;
    b(1, 0) = 4;
    b(1, 1) = 5;
    b(2, 0) = 6;
    b(2, 1) = 7;

    c(0, 0) = 3;
    c(0, 1) = 5;
    c(1, 0) = 7;
    c(1, 1) = 9;
    c(2, 0) = 11;
    c(2, 1) = 13;

    lucius::lazy::newThreadLocalContext();

    LazyValue lazyA = lucius::lazy::getConstant(a);
    LazyValue lazyB = lucius::lazy::getConstant(b);

    auto computed = lucius::lazy::applyBinary(lazyA, lazyB, lucius::lazy::Add()).materialize();

    if(computed != c)
    {
        lucius::util::log("test-lazy-ir") << " Matrix Addition Test Failed:\n";
        lucius::util::log("test-lazy-ir") << "  result matrix " << computed.toString();
        lucius::util::log("test-lazy-ir") << "  does not match reference matrix " << c.toString();
    }
    else
    {
        lucius::util::log("test-lazy-ir") << " Matrix Addition Test Passed\n";
    }

    return computed == c;
}

/*
    Test matrix addition loop

    a = [ 0 0 ]
        [ 0 0 ]
        [ 0 0 ]

    b = [ 1 4 ]
        [ 2 5 ]
        [ 3 6 ]

    for (i=0; i < 2; ++i)
    {
        a = a + b;
    }

    a = [ 2  8 ]
        [ 4 10 ]
        [ 6 12 ]
*/
static bool testLoop()
{
    Matrix a(3, 2);
    Matrix b(3, 2);
    Matrix c(3, 2);

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;
    b(2, 0) = 5;
    b(2, 1) = 6;

    c(0, 0) = 2;
    c(0, 1) = 4;
    c(1, 0) = 6;
    c(1, 1) = 8;
    c(2, 0) = 10;
    c(2, 1) = 12;

    lucius::lazy::newThreadLocalContext();

    LazyValue lazyA = lucius::lazy::zeros(a.size(), a.precision());
    LazyValue lazyB = lucius::lazy::getConstant(b);

    lucius::lazy::forLoop(2, [=]()
    {
        lucius::lazy::copy(lazyA, lucius::lazy::applyBinary(lazyA, lazyB, lucius::lazy::Add()));
    });

    auto computed = lazyA.materialize();

    if(computed != c)
    {
        lucius::util::log("test-lazy-ir") << " Matrix Addition Loop Test Failed:\n";
        lucius::util::log("test-lazy-ir") << "  result matrix " << computed.toString();
        lucius::util::log("test-lazy-ir") << "  does not match reference matrix " << c.toString();
    }
    else
    {
        lucius::util::log("test-lazy-ir") << " Matrix Addition Loop Test Passed\n";
    }

    return computed == c;
}

/*
    Test random matrix creation

    a = randInit(0)
    b = randInit(1)

    assert a != b
*/
static bool testRandom()
{
    lucius::lazy::newThreadLocalContext();

    LazyValue lazyA = lucius::lazy::createInitializer([]()
    {
        LazyValue randomState = lucius::lazy::srand(0);

        return lucius::lazy::rand(randomState, {3, 2}, SinglePrecision());
    });

    LazyValue lazyB = lucius::lazy::createInitializer([]()
    {
        LazyValue randomState = lucius::lazy::srand(377);

        return lucius::lazy::rand(randomState, {3, 2}, SinglePrecision());
    });

    Matrix firstRun  = lazyA.materialize();
    Matrix secondRun = lazyB.materialize();

    if(firstRun == secondRun)
    {
        lucius::util::log("test-lazy-ir") << " Lazy IR Random Initialization Test Failed:\n";
        lucius::util::log("test-lazy-ir") << "  first run matrix " << firstRun.toString();
        lucius::util::log("test-lazy-ir") << "  does match second run matrix "
            << secondRun.toString();
    }
    else
    {
        lucius::util::log("test-lazy-ir") << " Lazy IR Random Initialization Test Passed\n";
    }

    return firstRun != secondRun;
}

/*
    Test matrix addition loop with an initializer

    a = randInit

    b = [ 1 4 ]
        [ 2 5 ]
        [ 3 6 ]

    for (i=0; i < 2; ++i)
    {
        a = a + b;
    }

    assert('a is initialized only once');
*/
static bool testLoopWithInitializer()
{
    Matrix b(3, 2);

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;
    b(2, 0) = 5;
    b(2, 1) = 6;

    lucius::lazy::newThreadLocalContext();

    LazyValue lazyB = lucius::lazy::getConstant(b);

    LazyValue lazyA = lucius::lazy::createInitializer([]()
    {
        LazyValue randomState = lucius::lazy::srand(177);

        return lucius::lazy::rand(randomState, {3, 2}, SinglePrecision());
    });

    size_t iterations = 2;

    lucius::lazy::forLoop(iterations, [=]()
    {
        lucius::lazy::copy(lazyA, lucius::lazy::applyBinary(lazyA, lazyB, lucius::lazy::Add()));
    });

    Matrix firstRun  = lazyA.materialize();
    Matrix secondRun = lazyA.materialize();

    if(firstRun != secondRun)
    {
        lucius::util::log("test-lazy-ir") << " Lazy IR Loop With Initialization Test Failed:\n";
        lucius::util::log("test-lazy-ir") << "  first run matrix " << firstRun.toString();
        lucius::util::log("test-lazy-ir") << "  does not match second run matrix "
            << secondRun.toString();
    }
    else
    {
        lucius::util::log("test-lazy-ir") << " Lazy IR Loop With Initialization Test Passed\n";
    }

    return firstRun == secondRun;
}

/*
    Test matrix addition loop with an initializer and update

    a = [ 0 0 ]
        [ 0 0 ]
        [ 0 0 ]

    b = [ 1 4 ]
        [ 2 5 ]
        [ 3 6 ]

    for (i=0; i < 2; ++i)
    {
        cost = reduce(a * b);
        a = a + dcost/da;
    }
*/
static bool testLoopWithInitializerAndUpdate()
{
    Matrix b(3, 2);

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;
    b(2, 0) = 5;
    b(2, 1) = 6;

    lucius::lazy::newThreadLocalContext();

    LazyValue lazyB = lucius::lazy::getConstant(b);

    LazyValue lazyA = lucius::lazy::createInitializer([]()
    {
        return lucius::lazy::zeros({3, 2}, SinglePrecision());
    });

    lucius::lazy::forLoop(2, [=]()
    {
        auto aTimesB = lucius::lazy::applyBinary(lazyA, lazyB, lucius::lazy::Multiply());

        auto cost = lucius::lazy::castToScalar(
            lucius::lazy::reduce(aTimesB, {}, lucius::lazy::Add()));

        auto variablesAndGradients = lucius::lazy::getVariablesAndGradientsForCost(cost);

        for(auto& variableAndGradient : variablesAndGradients)
        {
            lucius::lazy::copy(variableAndGradient.getVariable(),
                lucius::lazy::applyBinary(variableAndGradient.getVariable(),
                    variableAndGradient.getGradient(), lucius::lazy::Add()));
        }
    });

    auto computed = lazyA.materialize();

    Matrix c = apply(b, lucius::matrix::Multiply(2));

    if(computed != c)
    {
        lucius::util::log("test-lazy-ir") << " Lazy IR Loop With Initialization "
            "And Update Test Failed:\n";
        lucius::util::log("test-lazy-ir") << "  computed matrix " << computed.toString();
        lucius::util::log("test-lazy-ir") << "  does not match reference matrix "
            << c.toString();
    }
    else
    {
        lucius::util::log("test-lazy-ir") << " Lazy IR Loop With Initialization "
            "And Update Test Passed\n";
    }

    return computed == c;
}

/*
    Test matrix addition after saving loading

    [ 1 2 ] + [ 2 3 ] = [  3  5 ]
    [ 3 4 ]   [ 4 5 ]   [  7  9 ]
    [ 5 6 ]   [ 6 7 ]   [ 11 13 ]

*/
static bool testSaveAndLoad()
{
    Matrix a(3, 2);
    Matrix b(3, 2);
    Matrix c(3, 2);

    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    a(2, 0) = 5;
    a(2, 1) = 6;

    b(0, 0) = 2;
    b(0, 1) = 3;
    b(1, 0) = 4;
    b(1, 1) = 5;
    b(2, 0) = 6;
    b(2, 1) = 7;

    c(0, 0) = 3;
    c(0, 1) = 5;
    c(1, 0) = 7;
    c(1, 1) = 9;
    c(2, 0) = 11;
    c(2, 1) = 13;

    lucius::lazy::newThreadLocalContext();

    LazyValue lazyA = lucius::lazy::getConstant(a);
    LazyValue lazyB = lucius::lazy::getConstant(b);

    auto lazyComputed = lucius::lazy::applyBinary(lazyA, lazyB, lucius::lazy::Add());

    auto lazyComputedHandle = lucius::lazy::getHandle(lazyComputed);

    std::stringstream stream;

    lucius::lazy::saveThreadLocalContext(stream);
    lucius::lazy::newThreadLocalContext();
    lucius::lazy::loadThreadLocalContext(stream);

    lazyComputed = lucius::lazy::lookupValueByHandle(lazyComputedHandle);

    auto computed = lazyComputed.materialize();

    if(computed != c)
    {
        lucius::util::log("test-lazy-ir") << " Matrix Addition Test Failed:\n";
        lucius::util::log("test-lazy-ir") << "  result matrix " << computed.toString();
        lucius::util::log("test-lazy-ir") << "  does not match reference matrix " << c.toString();
    }
    else
    {
        lucius::util::log("test-lazy-ir") << " Matrix Addition Test Passed\n";
    }

    return computed == c;
}

static bool runTests(bool listTests, const std::string& testFilter)
{
    lucius::util::TestEngine engine;

    engine.addTest("add", testAdd);
    engine.addTest("loop", testLoop);
    engine.addTest("random", testRandom);
    engine.addTest("initializer loop", testLoopWithInitializer);
    engine.addTest("initializer and update loop", testLoopWithInitializerAndUpdate);
    engine.addTest("save load ", testSaveAndLoad);

    if(listTests)
    {
        std::cout << engine.listTests();

        return true;
    }
    else
    {
        return engine.run(testFilter);
    }
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    parser.description("Unit tests for lazy ir operations.");

    std::string loggingEnabledModules = "test-lazy-ir";

    bool verbose = false;
    bool listTests = false;
    std::string testFilter;

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution.");
    parser.parse("-l", "--list-tests", listTests, false,
        "List all possible tests.");
    parser.parse("-L", "--log-module", loggingEnabledModules, loggingEnabledModules,
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");
    parser.parse("-t", "--test-filter", testFilter, "",
        "Only run tests that match the regular expression.");
    parser.parse();

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::util::enableSpecificLogs(loggingEnabledModules);
    }

    lucius::util::log("test-lazry-ir") << "Running matrix unit tests\n";

    bool passed = runTests(listTests, testFilter);

    if(!passed)
    {
        std::cout << "Test Failed\n";
    }
    else
    {
        std::cout << "Test Passed\n";
    }

    return 0;
}

