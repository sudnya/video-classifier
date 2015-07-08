/*! \file   TestOptimizer.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Saturday September 28, 2014
    \brief  The source file for the TestOptimizer tool.
*/

// Lucious Includes
#include <lucious/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <lucious/optimizer/interface/GeneralDifferentiableSolver.h>
#include <lucious/optimizer/interface/CostAndGradientFunction.h>

#include <lucious/matrix/interface/Matrix.h>
#include <lucious/matrix/interface/MatrixVector.h>
#include <lucious/matrix/interface/MatrixVectorOperations.h>
#include <lucious/matrix/interface/Operation.h>

#include <lucious/util/interface/ArgumentParser.h>
#include <lucious/util/interface/Knobs.h>
#include <lucious/util/interface/string.h>
#include <lucious/util/interface/debug.h>

// Standard Library Includes
#include <iostream>
#include <cassert>
#include <cmath>

class SimpleQuadraticCostAndGradientFunction : public lucious::optimizer::CostAndGradientFunction
{
public:
    virtual double computeCostAndGradient(MatrixVector& gradient,
        const MatrixVector& inputs) const
    {
        double cost = lucious::matrix::dotProduct(inputs, inputs);

        gradient = apply(inputs, lucious::matrix::Multiply(2.0));

        return cost;
    }

};

static bool testSimpleQuadratic(const std::string& name)
{
    // Solves a quadratic (convex) in 16 dimensions

    auto solver = lucious::optimizer::GeneralDifferentiableSolverFactory::create(name);

    assert(solver != nullptr);

    lucious::matrix::Matrix input(4, 4);

    input(0, 0) = 0.5;
    input(0, 1) = 0.5;
    input(0, 2) = 0.5;
    input(0, 3) = 0.5;

    input(1, 0) = 0.5;
    input(1, 1) = 0.5;
    input(1, 2) = 0.5;
    input(1, 3) = 0.5;

    input(2, 0) = 0.5;
    input(2, 1) = 0.5;
    input(2, 2) = 0.5;
    input(2, 3) = 0.5;

    input(3, 0) = 0.5;
    input(3, 1) = 0.5;
    input(3, 2) = 0.5;
    input(3, 3) = 0.5;

    float finalCost = solver->solve(input, SimpleQuadraticCostAndGradientFunction());

    bool success = std::abs(finalCost) < 1.0e-3;

    if(success)
    {
        std::cout << "  Test Simple Quadratic Passed\n";
    }
    else
    {
        std::cout << "  Test Simple Quadratic Failed\n";
    }

    return success;
}

class SimpleQuarticCostAndGradientFunction : public lucious::optimizer::CostAndGradientFunction
{
public:
    virtual double computeCostAndGradient(MatrixVector& gradient,
        const MatrixVector& inputs) const
    {
        auto shiftedInputs = apply(inputs, lucious::matrix::Add(-3.0));

        auto shiftedInputsCubed = apply(shiftedInputs, lucious::matrix::Pow(3.0));

        double cost = reduce(apply(MatrixVector(shiftedInputsCubed), shiftedInputs, lucious::matrix::Multiply()), {}, lucious::matrix::Add())[0][0];

        gradient = apply(shiftedInputsCubed, lucious::matrix::Multiply(4.0));

        return cost;
    }

};

static bool testSimpleQuartic(const std::string& name)
{
    // Solves a quartic (convex) in 16 dimensions
    auto solver = lucious::optimizer::GeneralDifferentiableSolverFactory::create(name);

    assert(solver != nullptr);

    lucious::matrix::Matrix input(4, 4);

    input(0, 0) = 0.5;
    input(0, 1) = 0.5;
    input(0, 2) = 0.5;
    input(0, 3) = 0.5;

    input(1, 0) = 0.5;
    input(1, 1) = 0.5;
    input(1, 2) = 0.5;
    input(1, 3) = 0.5;

    input(2, 0) = 0.5;
    input(2, 1) = 0.5;
    input(2, 2) = 0.5;
    input(2, 3) = 0.5;

    input(3, 0) = 0.5;
    input(3, 1) = 0.5;
    input(3, 2) = 0.5;
    input(3, 3) = 0.5;

    float finalCost = solver->solve(input, SimpleQuarticCostAndGradientFunction());

    bool success = std::abs(finalCost) < 1.0e-3;

    if(success)
    {
        std::cout << "  Test Simple Quartic Passed\n";
    }
    else
    {
        std::cout << "  Test Simple Quartic Failed\n";
    }

    return success;
}


static bool test(const std::string& name)
{
    std::cout << "Testing solver '" << name << "'\n";

    bool success = true;

    success &= testSimpleQuadratic(name);
    success &= testSimpleQuartic(name);

    if(success)
    {
        std::cout << " Solver '" << name << "' Passed All Tests\n";
    }
    else
    {
        std::cout << " Solver '" << name << "' Failed Some Tests\n";
    }

    return success;
}

static void setupSolverParameters()
{
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "3.0e-2");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9999");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.000");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "2000.0");
    lucious::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1000");
}

int main(int argc, char** argv)
{
    lucious::util::ArgumentParser parser(argc, argv);

    bool enableAllLogs = false;
    std::string logs = "";

    parser.parse("-l", "--enable-all-logs", enableAllLogs, false, "Enable all logs.");
    parser.parse("-L", "--enable-logs", logs, "", "Enable the specified logs (comma-separated).");

    parser.parse();

    if(enableAllLogs)
    {
        lucious::util::enableAllLogs();
    }

    lucious::util::enableSpecificLogs(logs);

    setupSolverParameters();

    auto solvers = lucious::optimizer::GeneralDifferentiableSolverFactory::enumerate();

    bool success = true;

    for(auto solver : solvers)
    {
        success &= test(solver);
    }

    if(success)
    {
        std::cout << "Test Passed\n";
    }
    else
    {
        std::cout << "Test Failed\n";
    }

    return success ? 0 : -1;
}




