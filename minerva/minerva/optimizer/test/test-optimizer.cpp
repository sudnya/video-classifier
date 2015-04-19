/*! \file   TestOptimizer.cpp
	\author Gregory Diamos <gregory.diamos@gmail.com>
	\date   Saturday September 28, 2014
	\brief  The source file for the TestOptimizer tool.
*/

// Minerva Includes
#include <minerva/optimizer/interface/GeneralDifferentiableSolverFactory.h>
#include <minerva/optimizer/interface/GeneralDifferentiableSolver.h>
#include <minerva/optimizer/interface/CostAndGradientFunction.h>

#include <minerva/matrix/interface/Matrix.h>
#include <minerva/matrix/interface/MatrixVector.h>
#include <minerva/matrix/interface/MatrixVectorOperations.h>
#include <minerva/matrix/interface/Operation.h>

#include <minerva/util/interface/ArgumentParser.h>
#include <minerva/util/interface/Knobs.h>
#include <minerva/util/interface/string.h>
#include <minerva/util/interface/debug.h>

// Standard Library Includes
#include <iostream>
#include <cassert>
#include <cmath>

class SimpleQuadraticCostAndGradientFunction : public minerva::optimizer::CostAndGradientFunction
{
public:
	virtual double computeCostAndGradient(MatrixVector& gradient,
		const MatrixVector& inputs) const
	{
		double cost = minerva::matrix::dotProduct(inputs, inputs);

		gradient = apply(inputs, minerva::matrix::Multiply(2.0));

		return cost;
	}

};

static bool testSimpleQuadratic(const std::string& name)
{
	// Solves a quadratic (convex) in 16 dimensions

	auto solver = minerva::optimizer::GeneralDifferentiableSolverFactory::create(name);

	assert(solver != nullptr);

	minerva::matrix::Matrix input(4, 4);

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

class SimpleQuarticCostAndGradientFunction : public minerva::optimizer::CostAndGradientFunction
{
public:
	virtual double computeCostAndGradient(MatrixVector& gradient,
		const MatrixVector& inputs) const
	{
		auto shiftedInputs = apply(inputs, minerva::matrix::Add(-3.0));

		auto shiftedInputsCubed = apply(shiftedInputs, minerva::matrix::Pow(3.0));

		double cost = reduce(apply(MatrixVector(shiftedInputsCubed), shiftedInputs, minerva::matrix::Multiply()), {}, minerva::matrix::Add())[0][0];

		gradient = apply(shiftedInputsCubed, minerva::matrix::Multiply(4.0));

		return cost;
	}

};

static bool testSimpleQuartic(const std::string& name)
{
	// Solves a quartic (convex) in 16 dimensions
	auto solver = minerva::optimizer::GeneralDifferentiableSolverFactory::create(name);

	assert(solver != nullptr);

	minerva::matrix::Matrix input(4, 4);

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
	minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::LearningRate", "3.0e-2");
	minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::Momentum", "0.9999");
	minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::AnnealingRate", "1.000");
	minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::MaxGradNorm", "2000.0");
	minerva::util::KnobDatabase::setKnob("NesterovAcceleratedGradient::IterationsPerBatch", "1000");
}

int main(int argc, char** argv)
{
	minerva::util::ArgumentParser parser(argc, argv);

	bool enableAllLogs = false;
	std::string logs = "";

	parser.parse("-l", "--enable-all-logs", enableAllLogs, false, "Enable all logs.");
	parser.parse("-L", "--enable-logs", logs, "", "Enable the specified logs (comma-separated).");

	parser.parse();

	if(enableAllLogs)
	{
		minerva::util::enableAllLogs();
	}

	minerva::util::enableSpecificLogs(logs);

	setupSolverParameters();

	auto solvers = minerva::optimizer::GeneralDifferentiableSolverFactory::enumerate();

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




