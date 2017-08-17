/*  \file   test-lazy-ir.cpp
    \date   April 17, 2017
    \author Gregory Diamos
    \brief  The source file for the lazy ir unit tests class.
*/

/*
    Test matrix addition

    [ 1 2 ] + [ 2 3 ] = [  3  5 ]
    [ 3 4 ]   [ 4 5 ]   [  7  9 ]
    [ 5 6 ]   [ 6 7 ]   [ 11 13 ]

*/
static void testAdd()
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

    lazy::newThreadLocalContext();

    LazyValue lazyA = lazy::getConstant(a);
    LazyValue lazyB = lazy::getConstant(b);

    auto computed = lazy::applyBinary(lazyA, lazyB, lazy::Add()).materialize<Matrix>();

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
static void testLoop()
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

    c(0, 0) = 3;
    c(0, 1) = 5;
    c(1, 0) = 7;
    c(1, 1) = 9;
    c(2, 0) = 11;
    c(2, 1) = 13;

    lazy::newThreadLocalContext();

    LazyValue lazyA = lazy::getConstant(a);
    LazyValue lazyB = lazy::getConstant(b);

    lazy::zeros(lazyA);

    lazy::forLoop(2, [=]()
    {
        lazy::copy(lazyA, lazy::applyBinary(lazyA, lazyB, lazy::Add()));
    });

    auto computed = lazyA.materialize<Matrix>();

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
static void testLoopWithInitializer()
{
    Matrix b(3, 2);

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;
    b(2, 0) = 5;
    b(2, 1) = 6;

    lazy::newThreadLocalContext();

    LazyValue lazyB = lazy::getConstant(b);

    LazyValue lazyA = lazy::createInitializer([]()
    {
        LazyValue randomState = lazy::srand(177);

        return lazy::rand(randomState, {3, 2}, SinglePrecision());
    });

    lazy::forLoop(2, [&]()
    {
        lazy::copy(lazyA, lazy::applyBinary(lazyA, lazyB, lazy::Add()));
    });

    Matrix firstRun  = lazyA.materialize<Matrix>();
    Matrix secondRun = lazyA.materialize<Matrix>();

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
static void testLoopWithInitializerAndUpdate()
{
    Matrix b(3, 2);

    b(0, 0) = 1;
    b(0, 1) = 2;
    b(1, 0) = 3;
    b(1, 1) = 4;
    b(2, 0) = 5;
    b(2, 1) = 6;

    lazy::newThreadLocalContext();

    LazyValue lazyB = lazy::getConstant(b);

    LazyValue lazyA = lazy::createVariableInitializer([]()
    {
        return lazy::zeros({3, 2}, SinglePrecision());
    });

    lazy::forLoop(2, [&]()
    {
        auto aTimesB = lazy::applyBinary(lazyA, lazyB, lazy::Multiply())

        auto cost = lazy::castToScalar(lazy::reduce(aTimesB, {}, lazy::Add()));

        auto variablesAndGradients = lazy::getVariablesAndGradientsForCost(cost);

        for(auto& variableAndGradient : variablesAndGradients)
        {
            lazy::copy(variableAndGradient.getVariable(),
                lazy::add(variableAndGradient.getVariable(), variableAndGradient.getGradient()));
        }
    });

    auto computed = lazyA.materialize<Matrix>();

    Matrix c = apply(b, matrix::Multiply(2));

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
static void testAdd()
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

    lazy::newThreadLocalContext();

    LazyValue lazyA = lazy::getConstant(a);
    LazyValue lazyB = lazy::getConstant(b);

    auto lazyComputed = lazy::applyBinary(lazyA, lazyB, lazy::Add());

    auto lazyComputedHandle = lazy::getHandle(lazyComputed);

    std::stringstream stream;

    lazy::saveThreadLocalContext(stream);
    lazy::newThreadLocalContext();
    lazy::loadThreadLocalContext(stream);

    lazyComputed = lazy::lookupValueByHandle(lazyComputedHandle);

    auto computed = lazyComputed.materialize<Matrix>();

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

