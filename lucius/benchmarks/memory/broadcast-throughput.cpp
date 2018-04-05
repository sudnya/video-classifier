/*! \file   broadcast-throughput.cpp
    \date   Tuesday June 2, 2015
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \brief  A benchmark for broadcast operation throughput.
*/

// Lucious Includes
#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixOperations.h>
#include <lucius/matrix/interface/GenericOperators.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Timer.h>

// Type definitions
typedef lucius::matrix::Matrix Matrix;

class Parameters
{
public:
    size_t rows;
    size_t columns;
    size_t iterations;

public:
    Parameters()
    {

    }
};

static void runTest(const Parameters& parameters)
{
    Matrix left({parameters.rows, parameters.columns});
    Matrix right({parameters.rows});

    Matrix result(left.size());

    // warm-up
    broadcast(result, left, right, {1}, lucius::matrix::Subtract());

    lucius::util::Timer timer;

    timer.start();

    for(size_t i = 0; i < parameters.iterations; ++i)
    {
        broadcast(result, left, right, {1}, lucius::matrix::Subtract());
    }

    timer.stop();

    double bytes = 3 * left.precision().size() * left.elements();
    double rate = bytes / (timer.seconds() * 1.0e9);

    std::cout << "Throughput is " << rate << " GB/s\n";
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    Parameters parameters;

    std::string loggingEnabledModules;
    bool verbose = false;

    parser.description("A test for lucius broadcast operation performance.");

    parser.parse("-L", "--log-module", loggingEnabledModules, "",
        "Print out log messages during execution for specified modules "
        "(comma-separated list of modules, e.g. NeuralNetwork, Layer, ...).");

    parser.parse("-i", "--iterations", parameters.iterations, 100,
        "The number of iterations to time.");
    parser.parse("-r", "--rows", parameters.rows, 64,
        "The number of rows in the broadcast.");
    parser.parse("-c", "--columns", parameters.columns, 262144,
        "The number of columns in the broadcast.");

    parser.parse("-v", "--verbose", verbose, false,
        "Print out log messages during execution");

    parser.parse();

    if(verbose)
    {
        lucius::util::enableAllLogs();
    }
    else
    {
        lucius::util::enableSpecificLogs(loggingEnabledModules);
    }

    try
    {
        runTest(parameters);
    }
    catch(const std::exception& e)
    {
        std::cout << "Lucius Broadcast Throughput Benchmark Failed:\n";
        std::cout << "Message: " << e.what() << "\n\n";
    }

    return 0;
}






