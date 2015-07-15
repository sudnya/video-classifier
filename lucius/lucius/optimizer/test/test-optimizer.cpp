/*! \file   TestOptimizer.cpp
    \author Gregory Diamos <gregory.diamos@gmail.com>
    \date   Saturday September 28, 2014
    \brief  The source file for the TestOptimizer tool.
*/

// Lucius Includes
#include <lucius/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <lucius/optimizer/interface/GeneralDifferentiableSolver.h>
#include <lucius/optimizer/interface/CostAndGradientFunction.h>

#include <lucius/matrix/interface/Matrix.h>
#include <lucius/matrix/interface/MatrixVector.h>
#include <lucius/matrix/interface/MatrixVectorOperations.h>
#include <lucius/matrix/interface/Operation.h>

#include <lucius/util/interface/ArgumentParser.h>
#include <lucius/util/interface/Knobs.h>
#include <lucius/util/interface/string.h>
#include <lucius/util/interface/debug.h>

// Standard Library Includes
#include <iostream>
#include <cassert>
#include <cmath>

class SimpleQuadraticCostAndGradientFunction : public lucius::optimizer::CostAndGradientFunction
{
public:
    virtual double computeCostAndGradient(MatrixVector& gradient,
        const MatrixVector& inputs) const
    {
        double cost = lucius::matrix::dotProduct(inputs, inputs);

        gradient = apply(inputs, lucius::matrix::Multiply(2.0));

        return cost;
    }

};

static bool testSimpleQuadratic(const std::string& name)
{
    // Solves a quadratic (convex) in 16 dimensions

    auto solver = lucius::optimizer::GeneralDifferentiableSolverFactory::create(name);

    assert(solver != nullptr);

    lucius::matrix::Matrix input(4, 4);

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

class SimpleQuarticCostAndGradientFunction : public lucius::optimizer::CostAndGradientFunction
{
public:
    virtual double computeCostAndGradient(MatrixVector& gradient,
        const MatrixVector& inputs) const
    {
        auto shiftedInputs = apply(inputs, lucius::matrix::Add(-3.0));

        auto shiftedInputsCubed = apply(shiftedInputs, lucius::matrix::Pow(3.0));

        double cost = reduce(apply(MatrixVector(shiftedInputsCubed), shiftedInputs, lucius::matrix::Multiply()), {}, lucius::matrix::Add())[0][0];

        gradient = apply(shiftedInputsCubed, lucius::matrix::Multiply(4.0));

        return cost;
    }

};

static bool testSimpleQuartic(const std::string& name)
{
    // Solves a quartic (convex) in 16 dimensions
    auto solver = lucius::optimizer::GeneralDifferentiableSolverFactory::create(name);

    assert(solver != nullptr);

    lucius::matrix::Matrix input(4, 4);

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
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "3.0e-2");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9999");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.000");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "2000.0");
    lucius::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1000");
}

int main(int argc, char** argv)
{
    lucius::util::ArgumentParser parser(argc, argv);

    bool enableAllLogs = false;
    std::string logs = "";

    parser.parse("-l", "--enable-all-logs", enableAllLogs, false, "Enable all logs.");
    parser.parse("-L", "--enable-logs", logs, "", "Enable the specified logs (comma-separated).");

    parser.parse();

    if(enableAllLogs)
    {
        lucius::util::enableAllLogs();
    }

    lucius::util::enableSpecificLogs(logs);

    setupSolverParameters();

    auto solvers = lucius::optimizer::GeneralDifferentiableSolverFactory::enumerate();

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




